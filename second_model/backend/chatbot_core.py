# backend/chatbot_core.py
import asyncio
import logging
import os
import sys
import json
import re 
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage 

try:
    from initial_interaction_handler import handle_initial_query
    from session_flow_handler import handle_ongoing_session_turn
    
    from groq_api import (
        call_groq_llm_final_answer_lc, 
        translate_text_lc
    )
    from vector_search import load_data, create_faiss_index
    from session_manager import ChatSession 
    from language_handler import (
        detect_language_and_intent, 
        get_language_name, 
        get_localized_keywords, 
        translate_english_to_darija_via_service
    )
    from knowledge_handler import handle_general_knowledge_query 
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in chatbot_core.py: {e}. Application will likely fail.", file=sys.stderr)
    sys.exit(1)

log = logging.getLogger(__name__)

DATA_FILE_NAME = os.getenv("RAG_DATA_FILE", "data.json")
COMPONENTS_DATA_FILE_NAME = os.getenv("COMPONENTS_DATA_FILE", "key_components.json")
IMAGE_BASE_PATH_USER_MSG = "troubleshooting/" 

data_store = None
index_store = None
text_to_original_data_idx_map_store = None
components_data_store: list = [] 
is_core_initialized = False

def initialize_chatbot_core():
    # ... (This function remains the same as your last full working version) ...
    global data_store, index_store, text_to_original_data_idx_map_store, components_data_store
    global is_core_initialized, DATA_FILE_NAME, COMPONENTS_DATA_FILE_NAME
    if is_core_initialized: return True
    load_dotenv() 
    log.info("--- Initializing Chatbot Core System ---")
    current_module_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_module_dir, DATA_FILE_NAME)
    components_data_file_path = os.path.join(current_module_dir, COMPONENTS_DATA_FILE_NAME)
    if not os.getenv("GROQ_API_KEY"): log.critical("CRITICAL: GROQ_API_KEY is not set.")
    if not os.path.exists(data_file_path): log.critical(f"CRITICAL: RAG data file '{data_file_path}' missing."); return False
    data_store = load_data(data_file_path)
    if not data_store: log.critical(f"CRITICAL: No RAG data loaded from {data_file_path}."); return False
    log.info(f"Loaded {len(data_store)} RAG entries from {data_file_path}.")
    index_store, text_to_original_data_idx_map_store = create_faiss_index(data_store)
    if not index_store: log.critical("CRITICAL: FAISS RAG index creation failed."); return False
    log.info("FAISS RAG index created successfully.")
    if os.path.exists(components_data_file_path):
        try:
            with open(components_data_file_path, 'r', encoding='utf-8') as f:
                components_data_store = json.load(f)
                if not isinstance(components_data_store, list): components_data_store = []
            log.info(f"Loaded {len(components_data_store)} component entries.")
        except Exception as e: log.error(f"Failed to load components data: {e}", exc_info=True); components_data_store = []
    else: components_data_store = []; log.warning(f"Components data file '{components_data_file_path}' not found.")
    is_core_initialized = True
    log.info("--- Chatbot Core Initialization Complete ---")
    return True


async def process_user_turn(session: ChatSession, user_input_raw: str) -> str:
    if not is_core_initialized:
        # ... (Error handling for uninitialized core) ...
        return "System is currently unavailable. Please try again shortly."
    
    # 1. Language Detection and Explicit Switch
    # ... (This section remains the same as your last full working version) ...
    detected_lang_code, detected_dialect_req_type, _ = await detect_language_and_intent(user_input_raw)
    is_explicit_lang_request = detected_dialect_req_type and "_request" in detected_dialect_req_type
    if is_explicit_lang_request:
        if detected_lang_code != session.current_language or \
           (detected_dialect_req_type and detected_dialect_req_type != session.last_detected_dialect_info): 
            session.set_language(detected_lang_code, detected_dialect_req_type)
            ack_ctx = (f"Language switched to {session.current_language_name} "
                       f"({session.last_detected_dialect_info or 'general'}). How can I help?")
            return await call_groq_llm_final_answer( # Use direct alias
                user_context_for_current_turn=ack_ctx,
                target_language_name=session.current_language_name,
                dialect_context_hint=session.last_detected_dialect_info,
                memory_messages=session.get_lc_memory_messages(),
                system_prompt_template_str="Respond in {{target_language_name}}."
            ) or f"Language set to {session.current_language_name}."
    elif detected_lang_code != session.current_language and not session.get_expectation():
        session.set_language(detected_lang_code, detected_dialect_req_type)


    # 2. Handle simple commands
    # ... (This section for /loadpdf, /clearpdf remains the same) ...
    if user_input_raw.lower().startswith("/loadpdf "): 
        return await call_groq_llm_final_answer("To upload a PDF, use the attachment button.", session.current_language_name, session.last_detected_dialect_info, session.get_lc_memory_messages()) or "Use attachment button for PDF."
    if user_input_raw.lower() == "/clearpdf":
        # ... (clear PDF logic) ...
        return await call_groq_llm_final_answer(f"PDF context cleared.", session.current_language_name, session.last_detected_dialect_info, session.get_lc_memory_messages()) or "PDF context cleared."


    # 3. Check for session reset/closing remarks
    # ... (This section remains the same as your last full working version) ...
    reset_kws = get_localized_keywords("session_reset_keywords", session.current_language)
    # ... (rest of reset logic) ...
    if not session.get_expectation() and (any(k.lower() in user_input_raw.lower() for k in reset_kws if k) or \
                                          any(k.lower() == user_input_raw.lower() for k in get_localized_keywords("simple_closing_remarks", session.current_language) if k)):
        # ... (construct reset_ctx_parts_english and system_prompt_for_closing) ...
        # ... (call session.end_session()) ...
        # ... (return localized closing response) ...
        log.info(f"CORE_PROCESS: Session reset/closing due to: '{user_input_raw[:50]}...'")
        # Simplified example for brevity
        session.end_session(reason=f"user_reset_or_closing: {user_input_raw.lower()}")
        return await call_groq_llm_final_answer("Okay, session ended. How can I help you next?", session.current_language_name, session.last_detected_dialect_info, []) or "Session ended."


    # 4. Main Handler Routing 
    intermediate_response_content: str | None = None 
    current_expectation = session.get_expectation()
    
    if current_expectation or session.in_troubleshooting_flow or session.active_tv_model:
        log.debug(f"CORE_PROCESS: Routing to handle_ongoing_session_turn. Expectation: {current_expectation}, Flow: {session.in_troubleshooting_flow}, ActiveModel: {session.active_tv_model}")
        intermediate_response_content = await handle_ongoing_session_turn(
            session=session, user_input_raw=user_input_raw,
            data_store=data_store, index_store=index_store,
            text_to_original_data_idx_map_store=text_to_original_data_idx_map_store,
            components_data_store=components_data_store, image_base_path=IMAGE_BASE_PATH_USER_MSG
        )
    else: 
        log.debug(f"CORE_PROCESS: Routing to handle_initial_query.")
        intermediate_response_content = await handle_initial_query(
            session=session, user_input_raw=user_input_raw,
            data_store=data_store, index_store=index_store,
            text_to_original_data_idx_map_store=text_to_original_data_idx_map_store,
            components_data_store=components_data_store, image_base_path=IMAGE_BASE_PATH_USER_MSG
        )

    # 5. Process INTENT_MARKERs and NEW_PROBLEM_SUGGESTION (if any) from intermediate_response_content
    final_assistant_response = None
    
    if intermediate_response_content:
        if intermediate_response_content.startswith("INTENT_MARKER:GENERAL_QUESTION_DETECTED\n"):
            log.info("CORE_PROCESS: Detected in-context general question.")
            english_gk_answer_with_marker = intermediate_response_content.split("\n", 1)
            english_gk_answer = english_gk_answer_with_marker[1] if len(english_gk_answer_with_marker) > 1 else "I can look that up."
            
            localized_gk_answer = english_gk_answer 
            if session.current_language != "en":
                translated = await translate_text_lc(english_gk_answer, "English", session.current_language_name, session.last_detected_dialect_info, "Answer to a general question.")
                if translated and not translated.startswith("Error:"): localized_gk_answer = translated
            
            offer_to_resume_en = ""
            if session.in_troubleshooting_flow and session.active_tv_model and session.current_problem_description:
                offer_to_resume_en = (f"\n\nNow, back to our troubleshooting for TV model '{session.active_tv_model}' "
                                      f"regarding '{session.current_problem_description[:50]}...'. Where were we, or what's the next step?")
            elif session.active_tv_model:
                 offer_to_resume_en = f"\n\nRegarding TV model '{session.active_tv_model}', is there anything specific you'd like to discuss or troubleshoot?"
            else:
                offer_to_resume_en = "\n\nIs there anything else I can help you with today?"

            localized_offer_to_resume = offer_to_resume_en
            if session.current_language != "en":
                 translated_offer = await translate_text_lc(offer_to_resume_en, "English", session.current_language_name, session.last_detected_dialect_info, "Offer to continue or ask for other help.")
                 if translated_offer and not translated_offer.startswith("Error:"): localized_offer_to_resume = translated_offer
            
            final_assistant_response = localized_gk_answer + localized_offer_to_resume
            # Session state (problem, model) is NOT cleared here.

        elif intermediate_response_content.startswith("INTENT_MARKER:PROBLEM_SOLVED\n"):
            log.info("CORE_PROCESS: Detected problem solved.")
            ack_message_en_with_marker = intermediate_response_content.split("\n", 1)
            ack_message_en = ack_message_en_with_marker[1] if len(ack_message_en_with_marker) > 1 else "Great to hear it's resolved!"
            
            localized_ack = ack_message_en
            if session.current_language != "en":
                # ... (translate ack_message_en to localized_ack) ...
                translated = await translate_text_lc(ack_message_en, "English", session.current_language_name, session.last_detected_dialect_info, "Acknowledgement of problem solved.")
                if translated and not translated.startswith("Error:"): localized_ack = translated

            ask_next_en = "\nWhat would you like to do next? Do you have another problem, or a general question?"
            localized_ask_next = ask_next_en
            if session.current_language != "en":
                # ... (translate ask_next_en to localized_ask_next) ...
                translated_ask = await translate_text_lc(ask_next_en, "English", session.current_language_name, session.last_detected_dialect_info, "Asking user for next action.")
                if translated_ask and not translated_ask.startswith("Error:"): localized_ask_next = translated_ask
            
            final_assistant_response = localized_ack + localized_ask_next
            session.clear_active_problem() 
            session.in_troubleshooting_flow = False 
        
        elif intermediate_response_content.startswith("INTENT_MARKER:PROBLEM_SOLVED_NEW_PROBLEM\n"):
            log.info("CORE_PROCESS: Detected problem solved AND new problem mentioned by LLM.")
            response_part_en_with_marker = intermediate_response_content.split("\n", 1)
            response_part_en = response_part_en_with_marker[1] if len(response_part_en_with_marker) > 1 else "Okay, what's the new problem?"

            localized_response = response_part_en
            if session.current_language != "en":
                # ... (translate response_part_en) ...
                translated = await translate_text_lc(response_part_en, "English", session.current_language_name, session.last_detected_dialect_info, "Acknowledging old problem solved and asking for new problem.")
                if translated and not translated.startswith("Error:"): localized_response = translated
            
            final_assistant_response = localized_response
            session.clear_active_problem() 
            session.in_troubleshooting_flow = False
            # The next user input will likely be handled by initial_interaction_handler to start the new problem.

        # Check for NEW_PROBLEM_SUGGESTION (already handled by session_flow_handler by setting expectation)
        # If the response from session_flow_handler *is* the NEW_PROBLEM_SUGGESTION confirmation question,
        # it will be passed through the normal localization below.
        # Example: "It sounds like... '{new_problem}'. Shall we focus on that? (yes/no)"

    # 6. Standard Localization if no special marker was fully processed above
    if final_assistant_response is None and intermediate_response_content:
        is_already_markdown_final = isinstance(intermediate_response_content, str) and \
                                     ("![alt text]" in intermediate_response_content or "## " in intermediate_response_content or "```" in intermediate_response_content) 
        is_error_msg_from_handler = isinstance(intermediate_response_content, str) and \
                                    intermediate_response_content.startswith("Error:")

        if session.current_language != "en" and \
           isinstance(intermediate_response_content, str) and \
           not is_already_markdown_final and \
           not is_error_msg_from_handler:
            
            if session.current_language == "ar" and \
               session.last_detected_dialect_info and "darija" in session.last_detected_dialect_info.lower():
                darija_translation = await translate_english_to_darija_via_service(intermediate_response_content)
                if darija_translation: final_assistant_response = darija_translation
                else: log.warning("CORE_PROCESS: Eng-to-Darija microservice failed. Using Groq LLM.")

            if not final_assistant_response: 
                localized_response = await translate_text_lc(
                    text_to_translate=intermediate_response_content,
                    source_language_name="English", target_language_name=session.current_language_name,
                    dialect_context_hint=session.last_detected_dialect_info,
                    context_hint_for_translation="Chatbot response."
                )
                if localized_response and not localized_response.startswith("Error:"):
                    final_assistant_response = localized_response
                else:
                    final_assistant_response = intermediate_response_content 
        else: 
             final_assistant_response = intermediate_response_content
    
    # 7. Final Fallback
    if not final_assistant_response:
         log.warning(f"CORE_PROCESS: No response after all processing for input '{user_input_raw[:50]}...'. Using LLM fallback.")
         fallback_context_english = "I'm not sure how to respond to that. Could you please rephrase your query, or let me know if you need help with a TV problem or have a general question?"
         final_assistant_response = await call_groq_llm_final_answer(
             user_context_for_current_turn=fallback_context_english,
             target_language_name=session.current_language_name,
             dialect_context_hint=session.last_detected_dialect_info,
             memory_messages=session.get_lc_memory_messages(),
             system_prompt_template_str="You are a helpful AI assistant. Respond in {{target_language_name}} based on the English fallback context. {{dialect_context_hint}}"
         )
         if not final_assistant_response or final_assistant_response.startswith("Error:"):
            log.error(f"CORE_PROCESS: LLM call for FINAL FALLBACK failed. Error: {final_assistant_response}")
            # Hardcoded fallback if LLM also fails for this.
            final_assistant_response = {"English": "I am currently unable to respond. Please try again later."}.get(session.current_language_name, "I am currently unable to respond. Please try again later.")


    return str(final_assistant_response)