import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime, timezone
from chatbot_logic import BomareChatbotAPIWrapper

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_very_secret_flask_key_123_XYZ')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, "expanded_data.json")
IMAGE_FOLDER_PATH = os.path.join(BASE_DIR, "images") 
STATIC_IMAGE_URL_PREFIX = "/static/bot_images"
BACKEND_BASE_URL = "http://localhost:5000"

try:
    if not os.path.exists(DATA_FILE_PATH): raise FileNotFoundError(f"Data file not found: {DATA_FILE_PATH}")
    if not os.path.isdir(IMAGE_FOLDER_PATH): print(f"Warning: Image folder not found: {IMAGE_FOLDER_PATH}.")
    chatbot_wrapper = BomareChatbotAPIWrapper(
        data_file_path=DATA_FILE_PATH,
        image_folder_path=IMAGE_FOLDER_PATH, 
        static_image_url_prefix=STATIC_IMAGE_URL_PREFIX,
        backend_base_url=BACKEND_BASE_URL
    )
except Exception as e:
    print(f"FATAL: Could not initialize BomareChatbotAPIWrapper: {e}")
    chatbot_wrapper = None 

chat_sessions = {} 

def get_or_create_session(session_id):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "id": session_id, "messages": [], "bot_state": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
            "title": f"Session {str(session_id)[:8]}"}
    return chat_sessions[session_id]

def update_session_title(session_id, first_user_message_text):
    session_data = get_or_create_session(session_id)
    if (session_data["title"].startswith("Session ") or session_data["title"].startswith("New Chat")) and first_user_message_text:
        words = first_user_message_text.split()
        new_title = " ".join(words[:5])
        if len(words) > 5: new_title += "..."
        if len(new_title) > 50: new_title = new_title[:47] + "..."
        session_data["title"] = new_title if new_title else f"Chat {str(session_id)[:4]}"

@app.route(f'{STATIC_IMAGE_URL_PREFIX}/<path:filename>')
def serve_bot_image(filename):
    return send_from_directory(IMAGE_FOLDER_PATH, filename)

@app.route('/api/chat_history', methods=['GET'])
def get_chat_history_list():
    history_summary = [{"id": sid, "title": sdata.get("title", f"Chat {sid[:8]}"), "last_message_timestamp": sdata.get("last_updated_at")}
                       for sid, sdata in sorted(chat_sessions.items(), key=lambda item: item[1].get("last_updated_at", "1970-01-01T00:00:00Z"), reverse=True)]
    return jsonify(history_summary)

@app.route('/api/new_chat', methods=['POST'])
def new_chat_session_api():
    session_id = str(uuid.uuid4())
    session_data = get_or_create_session(session_id) 
    session_data["title"] = f"New Chat {str(session_id)[:4]}" 
    session_data["messages"] = [] 
    session_data["bot_state"] = {} 
    return jsonify({"sessionId": session_id, "messages": session_data["messages"]})

@app.route('/api/chat_session/<session_id>', methods=['GET'])
def load_chat_session_messages_api(session_id):
    if session_id in chat_sessions:
        session_data = chat_sessions[session_id]
        return jsonify({"sessionId": session_id, "messages": session_data["messages"], "title": session_data.get("title", f"Session {session_id[:8]}")})
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/chat', methods=['POST'])
def handle_chat_message_api():
    if not chatbot_wrapper: return jsonify({"error": "Chatbot service is currently unavailable."}), 503
    form_data = request.form
    session_id = form_data.get('sessionId')
    user_message_text = form_data.get('message', '')
    language = form_data.get('language', 'en')
    mode = form_data.get('mode', 'Chatbot')
    uploaded_file = request.files.get('file')

    if not session_id: return jsonify({"error": "Session ID is required"}), 400
    if session_id not in chat_sessions:
         print(f"Warning: Received message for unknown session ID: {session_id}")
         return jsonify({"error": "Session not found or expired. Please start a new chat."}), 404

    session_data = get_or_create_session(session_id)
    user_msg_type = "text"
    actual_text_for_bot = user_message_text
    if uploaded_file:
        user_msg_type = "file"; user_display_text = uploaded_file.filename 
        if not actual_text_for_bot: actual_text_for_bot = f"User uploaded file: {uploaded_file.filename}"
    else:
        if not user_message_text: return jsonify({"error": "Empty message content"}), 400
        user_display_text = user_message_text
    user_msg_obj = {"sender": "user", "text": user_display_text, "timestamp": datetime.now(timezone.utc).isoformat(), "type": user_msg_type}
    if uploaded_file: user_msg_obj["originalFilename"] = uploaded_file.filename
    session_data["messages"].append(user_msg_obj)
    if sum(1 for m in session_data["messages"] if m["sender"] == "user") == 1:
        update_session_title(session_id, actual_text_for_bot)

    bot_responses_from_logic = chatbot_wrapper.process_message(actual_text_for_bot, session_data["bot_state"], language, mode)
    final_bot_replies_for_frontend = []
    for bot_resp_item in bot_responses_from_logic:
        reply_obj = {
            "sender": "bot", "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": bot_resp_item.get("type", "text"),
            # For text messages from logic, the text is in 'content'. For gallery, title is in 'text'.
            "text": bot_resp_item.get("content") if bot_resp_item.get("type") == "text" else bot_resp_item.get("text")
        }
        if reply_obj["type"] == "image":
            reply_obj["url"] = bot_resp_item.get("url")
            reply_obj["alt"] = bot_resp_item.get("alt")
            reply_obj["name"] = bot_resp_item.get("name")
            if not reply_obj.get("text") and (reply_obj.get("name") or reply_obj.get("alt")):
                reply_obj["text"] = reply_obj.get("name", reply_obj.get("alt", "Image")) 
        elif reply_obj["type"] == "image_gallery":
            reply_obj["items"] = bot_resp_item.get("items")
            if not reply_obj.get("text") and reply_obj.get("items"): # Fallback title
                reply_obj["text"] = f"Image Gallery ({len(reply_obj['items'])} images)"
        
        final_bot_replies_for_frontend.append(reply_obj)
        # Add to session_data["messages"] to store history
        # Ensure we are not adding a message that's already effectively there if API is called rapidly
        # (though this shouldn't happen with isLoadingResponse on frontend)
        is_duplicate = False
        for existing_msg in session_data["messages"]:
            if (existing_msg.get("sender") == "bot" and 
                existing_msg.get("type") == reply_obj.get("type") and
                existing_msg.get("text") == reply_obj.get("text")): # Simplified duplicate check
                # More robust would be to check timestamp proximity or unique ID from bot_logic if available
                is_duplicate = True
                break
        if not is_duplicate:
            session_data["messages"].append(reply_obj)


    session_data["last_updated_at"] = datetime.now(timezone.utc).isoformat()

    # --- ADDED LOGGING ---
    print("=" * 50)
    print(f"SENDING to frontend (session {session_id}):")
    for i, reply in enumerate(final_bot_replies_for_frontend):
        print(f"  Reply {i+1}: type='{reply.get('type')}', text='{reply.get('text')}'")
        if reply.get('type') == 'image_gallery':
            print(f"    Gallery items: {len(reply.get('items', []))}")
    print("=" * 50)
    # --- END OF ADDED LOGGING ---
    return jsonify({"replies": final_bot_replies_for_frontend})

if __name__ == '__main__':
    if not chatbot_wrapper: print("CRITICAL ERROR: Chatbot_wrapper failed to initialize.")
    else:
        print(f"Backend running at: {BACKEND_BASE_URL}")
        print(f"Image folder configured at: {os.path.abspath(IMAGE_FOLDER_PATH)}")
        print(f"Static image URL prefix for serving: {STATIC_IMAGE_URL_PREFIX}")
        print(f"Data file configured at: {os.path.abspath(DATA_FILE_PATH)}")
        app.run(debug=True, host='0.0.0.0', port=5000)