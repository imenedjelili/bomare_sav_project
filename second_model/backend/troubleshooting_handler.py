# backend/troubleshooting_handler.py
import logging
import sys 
import re 

try:
    from groq_api import (
        call_groq_llm_final_answer_lc as call_groq_llm_final_answer,
        generate_hypothetical_document_lc as generate_hypothetical_document,
        translate_text_lc as translate_input_for_rag,
    )
    from vector_search import search_relevant_guides 
    from session_manager import ChatSession 
    from knowledge_handler import handle_general_knowledge_query 
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in troubleshooting_handler.py: {e}. Application will likely fail.", file=sys.stderr)
    raise 

log = logging.getLogger(__name__)

async def handle_specific_tv_troubleshooting(
    user_problem_original_lang: str, 
    session: ChatSession, 
    data_store, index_store, text_to_original_data_idx_map_store,
    components_data_store, 
) -> str | None: 
    active_model = session.active_tv_model 
    if not active_model:
        log.error("TS_HANDLER_SPECIFIC: Called without an active TV model in session.")
        return ("I need to know which TV model you're referring to for specific troubleshooting. "
                "Could you please provide the model name?")

    log.info(f"TS_HANDLER_SPECIFIC: Starting specific troubleshooting. Model='{active_model}', "
             f"OriginalUserProblem='{user_problem_original_lang[:80]}...', UserLang='{session.current_language_name}'")

    if not index_store or not data_store:
        log.error("TS_HANDLER_SPECIFIC: RAG index or data_store is not available.")
        return "System error: The troubleshooting knowledge base is currently unavailable. Please try again later."

    problem_for_rag_en = user_problem_original_lang
    if session.current_language != "en":
        log.debug(f"TS_HANDLER_SPECIFIC: Translating problem to English for RAG. Original: '{user_problem_original_lang[:80]}'")
        translated_problem = await translate_input_for_rag(
            text_to_translate=user_problem_original_lang,
            source_language_name=session.current_language_name,
            target_language_name="English",
            dialect_context_hint=session.last_detected_dialect_info,
            context_hint_for_translation="TV problem description for troubleshooting lookup"
        )
        if translated_problem and not translated_problem.startswith("Error:"):
            problem_for_rag_en = translated_problem
            log.info(f"TS_HANDLER_SPECIFIC: Translated problem for RAG/HyDE: '{problem_for_rag_en[:80]}'")
        else:
            log.warning(f"TS_HANDLER_SPECIFIC: English translation of problem failed or empty. LLM response: {translated_problem}. "
                        f"Using original language input for RAG: '{user_problem_original_lang[:80]}'")
    else:
        log.debug(f"TS_HANDLER_SPECIFIC: Using original problem for RAG (already English): '{problem_for_rag_en[:80]}'")
            
    hypothetical_query_en = await generate_hypothetical_document(problem_for_rag_en)
    search_query_text_en = problem_for_rag_en 
    if hypothetical_query_en and not hypothetical_query_en.startswith("Error:"):
        search_query_text_en = hypothetical_query_en
        log.info(f"TS_HANDLER_SPECIFIC: Using HyDE-generated query for RAG: '{search_query_text_en[:80]}'")
    else:
        log.info(f"TS_HANDLER_SPECIFIC: HyDE failed or returned empty/error. "
                 f"Using translated/original problem as RAG query: '{search_query_text_en[:80]}'")

    NUMBER_OF_SEMANTIC_CANDIDATES_TO_CHECK = 5 
    log.debug(f"TS_HANDLER_SPECIFIC: Calling search_relevant_guides with: "
              f"query_text='{search_query_text_en[:80]}', target_model='{active_model}', "
              f"k_results={NUMBER_OF_SEMANTIC_CANDIDATES_TO_CHECK}")
    
    rag_result_guide_dict = search_relevant_guides(
        query_text=search_query_text_en, 
        target_model=active_model, 
        data=data_store,
        index=index_store, 
        text_to_original_data_idx_map=text_to_original_data_idx_map_store,
        k_results=NUMBER_OF_SEMANTIC_CANDIDATES_TO_CHECK
    )

    if rag_result_guide_dict:
        log.info(f"TS_HANDLER_SPECIFIC: RAG Found guide for model '{active_model}'. Issue='{rag_result_guide_dict.get('issue')}'")
        
        if rag_result_guide_dict.get("images"): session.current_model_general_images = rag_result_guide_dict.get("images")
        elif not session.current_model_general_images and active_model: 
            model_entry_for_images = next((item for item in data_store if item.get("model","").upper() == active_model.upper() and item.get("images")), None)
            if model_entry_for_images and model_entry_for_images.get("images"):
                session.current_model_general_images = model_entry_for_images.get("images")

        guide_issue_en = rag_result_guide_dict.get("issue", "the relevant troubleshooting information") 
        steps_en_list_raw = rag_result_guide_dict.get("steps")

        if not steps_en_list_raw or not isinstance(steps_en_list_raw, list) or \
           not any(s and isinstance(s, dict) and s.get("description") for s in steps_en_list_raw):
            log.warning(f"TS_HANDLER_SPECIFIC: RAG Guide '{guide_issue_en}' for model {active_model} has no valid raw steps.")
            return (f"I found information related to '{guide_issue_en}' for model '{active_model}', "
                    f"but it lacks detailed steps. Would you like general advice or to check for diagrams?")

        english_steps_for_llm_explanation = []
        for i, step_info in enumerate(steps_en_list_raw):
            if step_info and isinstance(step_info, dict):
                step_desc_en = step_info.get('description', '').strip()
                if step_desc_en:
                    english_steps_for_llm_explanation.append(f"{i+1}. {step_desc_en}")
        
        if not english_steps_for_llm_explanation:
            log.error(f"TS_HANDLER_SPECIFIC: No valid step descriptions extracted from raw steps for '{guide_issue_en}'.")
            return (f"I found a guide for '{guide_issue_en}' for model '{active_model}', "
                    f"but I couldn't prepare the steps for explanation. Try asking generally?")

        llm_explanation_context_en = f"""
        The user is troubleshooting their TV model '{active_model}' for an issue related to: "{guide_issue_en}".
        I have found the following troubleshooting steps from a technical guide:
        --- RAW STEPS START ---
        {chr(10).join(english_steps_for_llm_explanation)}
        --- RAW STEPS END ---

        Your task is to take these raw technical steps and explain them to a non-expert user in a clear, detailed, and user-friendly way. 
        For each step:
        1.  Restate the step's core action.
        2.  Elaborate on *why* this step is performed or what it helps to check.
        3.  Provide brief, simple guidance on *how* to perform the check if it's not obvious (e.g., "checking a cable means ensuring it's firmly plugged in at both ends and shows no visible damage").
        4.  Maintain a helpful and patient tone.
        5.  Use Markdown for formatting (e.g., numbered lists for the explained steps, bolding for emphasis).
        6.  Number your explained steps starting from 1.
        7.  After explaining all the steps, include the following "Important Safety Note" exactly as written:
            "Important Safety Note: Before attempting any internal checks or component replacements, please ensure your TV is completely unplugged from the power source. If you are unsure or uncomfortable with any step, it's always best to consult a qualified electronics technician."
        8.  Then, if any media (images, diagrams, component lists) might be relevant and available for this model (based on typical TV components), you can offer to show them. For example: "After you've reviewed these explanations, would you like to see if I have a motherboard image or a block diagram for your TV model {active_model}?" Do not list specific files, just general types of media.

        The final response you generate will be translated to the user's language if it's not English, so ensure your English explanation is clear and unambiguous.
        Do NOT add any conversational fluff before starting the step explanations, like "Okay, here are the explained steps:". Just start with the first explained step.
        """
        
        explained_steps_response_en = await call_groq_llm_final_answer(
            user_context_for_current_turn=llm_explanation_context_en,
            target_language_name="English", 
            dialect_context_hint=None, 
            memory_messages=session.get_lc_memory_messages(), 
            system_prompt_template_str=(
                "You are a helpful AI assistant that explains technical TV troubleshooting steps clearly to a non-expert user. "
                "You will be given raw steps and context. Your output should be a detailed, user-friendly explanation of these steps in {{target_language_name}} (which will be English for this call), "
                "followed by a safety note and an optional offer for related media. Use Markdown for formatting."
            )
        )

        if explained_steps_response_en and not explained_steps_response_en.startswith("Error:"):
            log.info(f"TS_HANDLER_SPECIFIC: LLM generated explained steps (English): '{explained_steps_response_en[:150]}...'")
            session.start_troubleshooting_flow(problem_for_rag_en, active_model) 
            return explained_steps_response_en 
        else:
            log.error(f"TS_HANDLER_SPECIFIC: LLM failed to explain steps for '{guide_issue_en}'. LLM response: {explained_steps_response_en}")
            raw_steps_formatted = "\n".join(english_steps_for_llm_explanation)
            safety_note = ("\n\n**Important Safety Note:** Before attempting any internal checks or component replacements, "
                           "please ensure your TV is completely unplugged from the power source. If you are unsure or uncomfortable "
                           "with any step, it's always best to consult a qualified electronics technician.")
            return (f"I found these steps for TV model '{active_model}' regarding '{guide_issue_en}':\n{raw_steps_formatted}"
                    f"{safety_note}\nI had trouble elaborating on them, but I hope this list helps.")
    else: 
        log.warning(f"TS_HANDLER_SPECIFIC: RAG Search did NOT find guide for model '{active_model}', query '{search_query_text_en}'.")
        offer_text_en = ""
        image_offers_en = [] 
        if session.current_model_general_images:
            if session.current_model_general_images.get('motherboard'): image_offers_en.append("a motherboard image")
            if session.current_model_general_images.get('key_components'): image_offers_en.append("a general key components image")
            if session.current_model_general_images.get('block_diagram'): image_offers_en.append("a block diagram")
        model_comp_data_entry = next((item for item in (components_data_store or []) if item.get("tv_model","").upper() == active_model.upper()), None)
        if model_comp_data_entry:
            if model_comp_data_entry.get("image_filename") and "a detailed key components diagram" not in image_offers_en:
                 image_offers_en.append("a detailed key components diagram")
            if model_comp_data_entry.get("key_components") and "a list of key components with descriptions" not in image_offers_en:
                image_offers_en.append("a list of key components with descriptions")
        if image_offers_en:
            types_str_en = ", ".join(sorted(list(set(image_offers_en))))
            offer_text_en = f" However, I can check if I have {types_str_en} for this model if you'd like."
        return (f"I searched for information on TV model '{active_model}' regarding an issue like '{problem_for_rag_en}', "
                f"but I couldn't find a specific troubleshooting guide for it in my current knowledge base. "
                f"Could you please try rephrasing the problem, or perhaps describe it in more detail?{offer_text_en}")

async def handle_standard_tv_troubleshooting(
    user_problem_original_lang: str, 
    session: ChatSession,
    ask_for_model_explicitly: bool = True 
) -> str | None: 
    log.info(f"TS_HANDLER_STANDARD: Generating generic TV advice for problem: '{user_problem_original_lang[:60]}...'. Ask for model: {ask_for_model_explicitly}")
    llm_context_for_general_advice_en = f"""
The user has described a TV problem: "{user_problem_original_lang}"
The specific TV model is NOT yet known.
Your task is to generate a helpful response in **English**. This response should:
1.  Acknowledge the user's problem briefly.
2.  Provide a list of general, common troubleshooting steps applicable to most TVs for such an issue.
    Consider steps like: power connections, input source, cables, restarting devices, remote control, ventilation.
3.  Include an "Important Safety Note"...
""" # Abridged for brevity, use full prompt from previous answer
    if ask_for_model_explicitly:
        llm_context_for_general_advice_en += (
            "\n4. **Crucially, at the very end... ask the user for their TV model number...**"
        ) 
    else:
        llm_context_for_general_advice_en += "\n4. Conclude by asking if these general steps help or if they have more details..."
    llm_context_for_general_advice_en += (
        "\nFormat your entire response in **English** using **Markdown**...\n"
        "Do NOT add any conversational fluff..."
    )
    system_prompt_template_en_advice = (
        "You are a helpful TV troubleshooting assistant... generate general advice in **English**..."
    ) 
    memory_for_this_call = session.get_lc_memory_messages()[:1] if session.get_lc_memory_messages() else []
    english_advice = await call_groq_llm_final_answer( 
        user_context_for_current_turn=llm_context_for_general_advice_en,
        target_language_name="English", dialect_context_hint=None,      
        memory_messages=memory_for_this_call, system_prompt_template_str=system_prompt_template_en_advice
    )
    if english_advice and not english_advice.startswith("Error:"):
        if ask_for_model_explicitly and \
           ("model number" not in english_advice.lower() and \
            "model of your tv" not in english_advice.lower()):
            model_request_fallback_en = (
                "\n\nTo help me provide more specific advice, could you please tell me the model number of your TV? ..."
            ) # Abridged
            english_advice += model_request_fallback_en
        session.start_troubleshooting_flow(user_problem_original_lang) 
        if ask_for_model_explicitly:
            session.set_expectation("model_for_problem", problem_context_for_model_request=user_problem_original_lang)
        return english_advice 
    else: 
        fallback_en = (
            f"I understand you're having an issue ... \"{user_problem_original_lang}\". Some general things to check..."
        ) # Abridged
        if ask_for_model_explicitly:
             fallback_en += ("\n\nTo help me provide more specific advice, could you please tell me the model number ...")
        session.start_troubleshooting_flow(user_problem_original_lang)
        if ask_for_model_explicitly:
            session.set_expectation("model_for_problem", problem_context_for_model_request=user_problem_original_lang)
        return fallback_en

async def handle_list_all_model_issues(session: ChatSession, data_store) -> str | None:
    active_model = session.active_tv_model
    if not active_model:
        log.warning("TS_HANDLER_LIST_ISSUES: No active TV model in session.")
        session.set_expectation("model_for_list_issues", details={"original_request": "list all issues"})
        return "I need to know which TV model you're interested in to list its issues. What is the model number?"
    log.info(f"TS_HANDLER_LIST_ISSUES: Preparing English list of issues for model {active_model}.")
    documented_issues_en = []
    if data_store:
        for item in data_store:
            if item.get("model", "").upper() == active_model.upper():
                issue_title_en = item.get("issue")
                if issue_title_en and isinstance(issue_title_en, str):
                    documented_issues_en.append(issue_title_en.strip())
    if documented_issues_en:
        unique_issues_en = sorted(list(set(documented_issues_en)))
        issues_list_str_md_en = "\n- ".join(unique_issues_en)
        return (
            f"For TV model '{active_model}', here are some of the documented issues I have information about:\n- {issues_list_str_md_en}\n\n"
            f"You can ask me for troubleshooting details on any specific issue from this list or about diagrams/components for this model."
        )
    else:
        return (
            f"I don't have a pre-compiled list of issues for TV model '{active_model}'. "
            f"If you describe a specific problem, I'll do my best to help, or you can ask about diagrams/components."
        )

async def handle_session_follow_up_llm_call(user_query_original_lang: str, session: ChatSession) -> str | None:
    if not session.active_tv_model and not session.in_troubleshooting_flow and not session.current_problem_description:
        log.warning("TS_HANDLER_FOLLOWUP_LLM: Called without sufficient context. Treating as general knowledge.")
        # This now calls handle_general_knowledge_query which might return localized Markdown or English
        # For consistency, this function aims to return English for chatbot_core to localize if needed.
        # So, we'd ideally get an English GK answer here.
        # For simplicity now, we'll let it pass through, chatbot_core will see if it needs localization.
        return await handle_general_knowledge_query(user_query_original_lang, session)

    log.info(f"TS_HANDLER_FOLLOWUP_LLM: Model '{session.active_tv_model or 'N/A'}', "
             f"Problem '{session.current_problem_description or 'General context'}'. Query: '{user_query_original_lang[:50]}...'")

    lc_memory_messages = session.get_lc_memory_messages() 
    pdf_context_str = session.get_pdf_context_for_llm(max_chars=500)
    last_bot_message_content = ""
    if lc_memory_messages:
        # Find the last AI message
        for msg in reversed(lc_memory_messages):
            if msg.type == "ai":
                last_bot_message_content = msg.content
                break
    
    context_for_llm_en = f"""
    The user is in an ongoing session, likely TV troubleshooting.
    - Current TV Model in Focus: {session.active_tv_model or 'Not specifically set yet.'}
    - Current Problem/Topic: {session.current_problem_description or ('General discussion about the TV model' if session.active_tv_model else 'General TV help or other topic.')}
    - My (AI Assistant's) previous response was: "{last_bot_message_content[:200].replace(chr(10), ' ')}..."
    - User's latest message (in their language): "{user_query_original_lang}"
    {pdf_context_str}

    Your task is to analyze the user's latest message and formulate a helpful **English** response.
    Also, determine the primary nature of their message:

    1.  **CONTINUE_TROUBLESHOOTING:** If they are asking for clarification on previous steps, reporting a step's outcome (worked/didn't work), or asking a direct question about the current troubleshooting steps or problem.
        - Action: Provide the helpful English response.

    2.  **GENERAL_QUESTION_UNRELATED_TO_STEPS:** If they ask a general knowledge question that is not directly about the current troubleshooting steps but might be related to a term used or a general TV concept (e.g., "what is a diode?", "how does HDMI work?").
        - Action: Respond with `INTENT_MARKER:GENERAL_QUESTION_DETECTED\n` followed by the English answer to their general question.

    3.  **PROBLEM_SOLVED:** If they indicate the current problem ('{session.current_problem_description or "the current issue"}') is now resolved (e.g., "it's working!", "fixed it, thanks", "that solved it").
        - Action: Respond with `INTENT_MARKER:PROBLEM_SOLVED\n` followed by a brief, positive English acknowledgment (e.g., "Great to hear it's resolved!").

    4.  **NEW_PROBLEM_MENTIONED:** If they clearly state the current problem is solved AND want to move to a *new, different problem*, OR if they ignore the current flow and introduce a new problem.
        - Action: If they explicitly say "problem solved, now I have an issue with X", use `INTENT_MARKER:PROBLEM_SOLVED_NEW_PROBLEM\n` followed by "Okay, glad the previous issue is solved. What's the new problem? If it's for a different TV model, please let me know."
        - If they just state a new problem without mentioning the old one, use the `NEW_PROBLEM_SUGGESTION: [Summary of new problem, max 15 words]. Model: [ModelIfMentionedOrCurrentActive]` token at the end of a brief English reply to their immediate query (as before).

    5.  **MEDIA_REQUEST:** If they are asking for media (images, diagrams, component lists) for the **active model**.
        - Action: Acknowledge this request clearly in English (e.g., "Okay, you're looking for the motherboard image for model {session.active_tv_model}. I will check on that."). The actual media retrieval is handled elsewhere based on this acknowledgment.

    Output format:
    - If one of the special INTENT_MARKERs is identified, start your response with it on a new line.
    - Otherwise, just provide the helpful English response for CONTINUE_TROUBLESHOOTING or MEDIA_REQUEST.
    - Use Markdown for formatting. Be concise.
    """
    
    system_prompt_en = (
        "You are an AI formulating an intermediate ENGLISH response in a TV troubleshooting chat. "
        "Analyze the user's input for specific intents like problem solved, a new general question, or a new problem. "
        "Use INTENT_MARKERs as instructed if such intents are detected. Otherwise, respond helpfully to continue the current flow. "
        "Your English output will be translated later if needed. Use Markdown."
    )

    english_response_llm = await call_groq_llm_final_answer(
        user_context_for_current_turn=context_for_llm_en,
        target_language_name="English", dialect_context_hint=None,
        memory_messages=lc_memory_messages,
        system_prompt_template_str=system_prompt_en
    )

    if english_response_llm and not english_response_llm.startswith("Error:"):
        log.info(f"TS_HANDLER_FOLLOWUP_LLM: Formulated English response (may include marker): '{english_response_llm[:150]}...'")
        return english_response_llm
    else:
        log.warning(f"TS_HANDLER_FOLLOWUP_LLM: LLM failed for follow-up. Query: '{user_query_original_lang[:50]}...' LLM_Error: {english_response_llm}")
        return (f"I'm having a bit of trouble with that follow-up for model '{session.active_tv_model or 'your TV'}' "
                f"regarding '{session.current_problem_description or 'our topic'}'. Could you rephrase?")