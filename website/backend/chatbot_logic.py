
# backend/chatbot_logic.py
import json
import re
import os
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator, LANGUAGES

# --- NLPHelper Class ---
class NLPHelper:
    def __init__(self):
        try:
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            # print(f"Error loading SentenceTransformer model: {e}")
            raise

    def find_best_match(self, description, candidates):
        if not description or not candidates:
            return None, 0.0
        try:
            description_embedding = self.model.encode(description, convert_to_tensor=True)
            candidates_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        except Exception as e:
            # print(f"Error encoding with SentenceTransformer: {e}")
            return None, 0.0

        similarities = util.cos_sim(description_embedding, candidates_embeddings)[0]
        best_index = similarities.argmax().item()
        return candidates[best_index], similarities[best_index].item()

# --- BomareChatbotAPIWrapper Class ---
class BomareChatbotAPIWrapper:
    MAX_STEPS_PER_BATCH = 4

    def __init__(self, data_file_path, image_folder_path, static_image_url_prefix="/static/bot_images", backend_base_url=""):
        try:
            with open(data_file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError:
            raise

        self.intents = self.data.get("chatbot", {}).get("intents", [])
        
        self.prompts_db = self.data.get("chatbot", {}).get("prompts_db", {})
        if not self.prompts_db: 
            self.raw_fallback_responses = self.data.get("chatbot", {}).get("fallback_responses", ["Sorry, I didn't understand."])
            self.prompts_db = {
                "welcome": { "en": "👋 Welcome to Bomare Technician Assistant!", "fr": "👋 Bienvenue chez l'Assistant Technicien Bomare !", "ar": "👋 مرحبًا بك في مساعد فني بومار !" },
                "how_can_i_help": { "en": "Hello! How can I help you with your Bomare TV today?", "fr": "Bonjour ! Comment puis-je vous aider avec votre téléviseur Bomare aujourd'hui ?", "ar": "مرحبًا! كيف يمكنني مساعدتك بخصوص تلفزيون بومار الخاص بك اليوم؟" },
                "model_problem_confirmation": {
                    "en": "Okay, so we're looking at a {} with an issue described as '{}'. Let's see what we can do.",
                    "fr": "D'accord, nous examinons donc un {} avec un problème décrit comme '{}'. Voyons ce que nous pouvons faire.",
                    "ar": "حسنًا، إذن نحن ننظر إلى {} مع مشكلة وُصفت بأنها '{}'. دعنا نرى ما يمكننا القيام به."
                },
                "model_known_ask_problem": {
                    "en": "I understand you're working with a {}. Could you please describe the problem you're facing?",
                    "fr": "Je comprends que vous travaillez avec un {}. Pourriez-vous s'il vous plaît décrire le problème que vous rencontrez ?",
                    "ar": "أفهم أنك تعمل على {}. هل يمكنك من فضلك وصف المشكلة التي تواجهها؟"
                },
                "problem_known_ask_model": {
                    "en": "I understand the issue is '{}'. Could you let me know which TV model this is for?",
                    "fr": "Je comprends que le problème est '{}'. Pourriez-vous me dire pour quel modèle de téléviseur cela concerne ?",
                    "ar": "أفهم أن المشكلة هي '{}'. هل يمكنك إخباري عن طراز التلفزيون المعني؟"
                },
                "prompt_model_not_detected": { "en": "I couldn't quite catch the TV model. Could you please provide the model name or number?", "fr": "Je n'ai pas bien saisi le modèle du téléviseur. Pourriez-vous s'il vous plaît fournir le nom ou le numéro du modèle ?", "ar": "لم أتمكن من تحديد طراز التلفزيون. هل يمكنك من فضلك تقديم اسم أو رقم الطراز؟" },
                "model_not_in_db": { "en": "Hmm, I don't seem to recognize that model. Could you double-check it or try a different model name?", "fr": "Hmm, je ne reconnais pas ce modèle. Pourriez-vous le vérifier à nouveau ou essayer un autre nom de modèle ?", "ar": "همم، لا يبدو أنني أتعرف على هذا الطراز. هل يمكنك التحقق منه مرة أخرى أو تجربة اسم طراز مختلف؟" },
                "no_flow_for_model": { "en": "Unfortunately, I don't have specific troubleshooting procedures for the model '{}' at the moment.", "fr": "Malheureusement, je n'ai pas de procédures de dépannage spécifiques pour le modèle '{}' pour le moment.", "ar": "للأسف، ليس لدي إجراءات محددة لاستكشاف الأخطاء وإصلاحها للطراز '{}' في الوقت الحالي." },
                "no_steps_for_problem": { "en": "For the model '{}' and the issue you described as '{}', I don't have detailed steps right now.", "fr": "Pour le modèle '{}' et le problème que vous avez décrit comme '{}', je n'ai pas d'étapes détaillées pour l'instant.", "ar": "بالنسبة للطراز '{}' والمشكلة التي وصفتها بأنها '{}'، ليس لدي خطوات مفصلة في الوقت الحالي." },
                "escalate_to_support_intro_options": {
                     "en": ["It seems we've tried all the available automated steps. If the issue persists, please provide more details below so our support team can assist you further.", "We've reached the end of my current suggestions. To get more help, please fill out the form to connect with our support specialists.", "I'm unable to resolve this with my current information. Please use the form below to submit your issue to our technical support team."],
                     "fr": ["Il semble que nous ayons essayé toutes les étapes automatisées disponibles. Si le problème persiste, veuillez fournir plus de détails ci-dessous afin que notre équipe d'assistance puisse vous aider davantage.", "Nous avons épuisé mes suggestions actuelles. Pour obtenir plus d'aide, veuillez remplir le formulaire pour contacter nos spécialistes de l'assistance.", "Je ne parviens pas à résoudre ce problème avec mes informations actuelles. Veuillez utiliser le formulaire ci-dessous pour soumettre votre problème à notre équipe de support technique."],
                     "ar": ["يبدو أننا جربنا جميع الخطوات الآلية المتاحة. إذا استمرت المشكلة، يرجى تقديم مزيد من التفاصيل أدناه حتى يتمكن فريق الدعم لدينا من مساعدتك بشكل أكبر.", "لقد وصلنا إلى نهاية اقتراحاتي الحالية. للحصول على مزيد من المساعدة، يرجى ملء النموذج للتواصل مع متخصصي الدعم لدينا.", "لا يمكنني حل هذه المشكلة بمعلوماتي الحالية. يرجى استخدام النموذج أدناه لإرسال مشكلتك إلى فريق الدعم الفني لدينا."]
                },
                "fallback_understand": { "en": self.raw_fallback_responses[0] if self.raw_fallback_responses else "I'm sorry, I didn't quite catch that. Could you rephrase?", "fr": "Je suis désolé, je n'ai pas bien compris. Pourriez-vous reformuler ?", "ar": "أنا آسف، لم أفهم ذلك جيدًا. هل يمكنك إعادة الصياغة؟" },
                "end_glad_to_help_options": { 
                    "en": ["Glad I could help! Have a great day.", "Happy to assist! Hope everything is running smoothly now.", "Awesome! Feel free to reach out if you need more help in the future.", "Great to hear it's sorted! Good luck with everything.", "Excellent! I'm here if you need anything else down the line."],
                    "fr": ["Ravi d'avoir pu aider ! Bonne journée.", "Content de vous avoir assisté ! J'espère que tout fonctionne bien maintenant.", "Super ! N'hésitez pas si vous avez besoin d'aide à nouveau à l'avenir.", "Ravi d'entendre que c'est réglé ! Bonne chance pour tout.", "Excellent ! Je suis là si vous avez besoin d'autre chose plus tard."],
                    "ar": ["سعدت بتقديم المساعدة! أتمنى لك يوماً رائعاً.", "يسرني أن أساعد! آمل أن كل شيء يعمل بسلاسة الآن.", "رائع! لا تتردد في التواصل إذا احتجت إلى مزيد من المساعدة في المستقبل.", "يسعدني سماع أن الأمر قد تم حله! حظا سعيدا في كل شيء.", "ممتاز! أنا هنا إذا احتجت إلى أي شيء آخر لاحقًا."]
                },
                "images_found_title_options": { 
                    "en": ["📷 Here are some visuals that might be helpful for the {} model:", "Take a look at these images related to the {}:", "I found some illustrations for the {} model you might find useful:", "Here are some relevant pictures for the {}:"],
                    "fr": ["📷 Voici quelques visuels qui pourraient être utiles pour le modèle {} :", "Jetez un œil à ces images relatives au {} :", "J'ai trouvé quelques illustrations pour le modèle {} qui pourraient vous être utiles :", "Voici quelques images pertinentes pour le {} :"],
                    "ar": ["📷 إليك بعض الصور التي قد تكون مفيدة لطراز {}:", "ألقِ نظرة على هذه الصور المتعلقة بـ {}:", "لقد وجدت بعض الرسوم التوضيحية لطراز {} قد تجدها مفيدة:", "إليك بعض الصور ذات الصلة بـ {}:"]
                },
                "no_images_found": { "en": "I couldn't find any specific images for the model {} right now.", "fr": "Je n'ai pas pu trouver d'images spécifiques pour le modèle {} pour l'instant.", "ar": "لم أتمكن من العثور على أي صور محددة للطراز {} في الوقت الحالي." },
                "steps_batch_intro_first_options": {
                    "en": ["Alright, let's try these initial steps. Take your time and follow along:", "Okay, here are the first few things to check. We'll go through them carefully:", "Let's begin with these checks. I'm here to guide you step-by-step:"],
                    "fr": ["Bien, essayons ces premières étapes. Prenez votre temps et suivez attentivement :", "D'accord, voici les premières choses à vérifier. Nous allons les parcourir attentivement :", "Commençons par ces vérifications. Je suis là pour vous guider pas à pas :"],
                    "ar": ["حسنًا، لنجرب هذه الخطوات الأولية. خذ وقتك واتبع التعليمات بعناية:", "تمام، إليك الأشياء الأولى التي يجب التحقق منها. سنتصفحها بعناية:", "لنبدأ بهذه الفحوصات. أنا هنا لإرشادك خطوة بخطوة:"]
                },
                "steps_batch_intro_next_options": {
                    "en": ["Okay, let's try the next set of steps. Hopefully, one of these will do the trick:", "Alright, moving on to the next suggestions. Let's see if these help:", "Let's continue with these potential solutions:"],
                    "fr": ["D'accord, essayons la série d'étapes suivante. Avec un peu de chance, l'une d'elles fonctionnera :", "Bien, passons aux suggestions suivantes. Voyons si celles-ci aident :", "Continuons avec ces solutions potentielles :"],
                    "ar": ["تمام، لنجرب المجموعة التالية من الخطوات. نأمل أن تنجح إحدى هذه الخطوات:", "حسنًا، ننتقل إلى الاقتراحات التالية. دعنا نرى إن كانت هذه ستساعد:", "لنستمر مع هذه الحلول المحتملة:"]
                },
                "step_connector_first_options": { "en": ["First,", "To start,", "Initially,", "Let's begin by"], "fr": ["Premièrement,", "Pour commencer,", "Initialement,", "Commençons par"], "ar": ["أولاً،", "للبدء،", "مبدئيًا،", "لنبدأ بـ"] },
                "step_connector_middle_options": { "en": ["Next,", "Then,", "After that,", "Also,", "Following that,"], "fr": ["Ensuite,", "Puis,", "Après cela,", "Aussi,", "Suite à cela,"], "ar": ["بعد ذلك،", "ثم،", "بعد ذلك،", "أيضًا،", "بعد ذلك،"] },
                "step_connector_final_options": { "en": ["Finally,", "Lastly,", "And for the last one in this batch,", "To wrap this up,"], "fr": ["Finalement,", "Enfin,", "Et pour le dernier de ce lot,", "Pour conclure,"], "ar": ["أخيرًا،", "في النهاية،", "وبالنسبة للخطوة الأخيرة في هذه المجموعة،", "لإنهاء هذا،"] },
                "resolved_prompt_options": { 
                    "en": ["Did that do the trick?", "Is everything working as expected now?", "Hopefully, that sorted things out. How is it looking?", "Let me know if that helped, or if we need to try something else.", "Any luck with those steps?", "How did it go? Is the issue resolved?"],
                    "fr": ["Est-ce que cela a fonctionné ?", "Est-ce que tout fonctionne comme prévu maintenant ?", "Espérons que cela a réglé le problème. Comment ça se présente ?", "Faites-moi savoir si cela a aidé, ou si nous devons essayer autre chose.", "Ces étapes ont-elles porté leurs fruits ?", "Comment cela s'est-il passé ? Le problème est-il résolu ?"],
                    "ar": ["هل نجحت هذه الطريقة؟", "هل كل شيء يعمل كما هو متوقع الآن؟", "نأمل أن يكون ذلك قد حل المشكلة. كيف تبدو الأمور؟", "أخبرني إذا كان ذلك مفيدًا، أو إذا كنا بحاجة إلى تجربة شيء آخر.", "هل كان هناك أي نجاح مع تلك الخطوات؟", "كيف سارت الامور؟ هل تم حل المشكلة؟"]
                }
            }
        else: 
            # Ensure new keys and old ones are correctly defaulted if prompts_db is loaded from JSON
            default_structure = {
                "welcome": { "en": "👋 Welcome to Bomare Technician Assistant!", "fr": "👋 Bienvenue chez l'Assistant Technicien Bomare !", "ar": "👋 مرحبًا بك في مساعد فني بومار !" },
                "how_can_i_help": { "en": "Hello! How can I help you with your Bomare TV today?", "fr": "Bonjour ! Comment puis-je vous aider avec votre téléviseur Bomare aujourd'hui ?", "ar": "مرحبًا! كيف يمكنني مساعدتك بخصوص تلفزيون بومار الخاص بك اليوم؟" },
                "model_problem_confirmation": {"en": "Okay, for your {} with the '{}' issue, let's proceed.", "fr": "Ok, pour votre {} avec le problème '{}', procédons.", "ar": "تمام، بالنسبة لجهاز {} الذي به مشكلة '{}'، دعنا نتابع."},
                "model_known_ask_problem": {"en": "Got it, model {}. What's the issue?", "fr": "Compris, modèle {}. Quel est le problème?", "ar": "فهمت، طراز {}. ما هي المشكلة؟"},
                "problem_known_ask_model": {"en": "Issue noted: '{}'. Which model is this?", "fr": "Problème noté: '{}'. Quel modèle est-ce?", "ar": "المشكلة ملحوظة: '{}'. ما هو هذا الطراز؟"},
                "prompt_model_not_detected": {"en": "Please specify the TV model.", "fr": "Veuillez spécifier le modèle.", "ar": "يرجى تحديد الطراز."},
                "model_not_in_db": {"en": "Model not recognized.", "fr": "Modèle non reconnu.", "ar": "الطراز غير موجود."},
                "no_flow_for_model": {"en": "No procedures for model '{}'.", "fr": "Pas de procédures pour '{}'.", "ar": "لا إجراءات للطراز '{}'."},
                "no_steps_for_problem": {"en": "No steps for '{}' with model '{}'.", "fr": "Pas d'étapes pour '{}' avec '{}'.", "ar": "لا خطوات لـ '{}' مع '{}'."},
                "fallback_understand": {"en": self.prompts_db.get("fallback_responses", ["Sorry, I didn't understand."])[0] if self.prompts_db.get("fallback_responses") else "I'm sorry, I didn't catch that.", "fr": "Je n'ai pas compris.", "ar": "لم أفهم."},
                "no_images_found": {"en": "No images for {}.", "fr": "Pas d'images pour {}.", "ar": "لا صور لـ {}."},
                "escalate_to_support_intro_options": {
                     "en": ["It seems we've tried all the available automated steps. If the issue persists, please provide more details below so our support team can assist you further."],
                     "fr": ["Il semble que nous ayons essayé toutes les étapes automatisées disponibles. Si le problème persiste, veuillez fournir plus de détails ci-dessous afin que notre équipe d'assistance puisse vous aider davantage."],
                     "ar": ["يبدو أننا جربنا جميع الخطوات الآلية المتاحة. إذا استمرت المشكلة، يرجى تقديم مزيد من التفاصيل أدناه حتى يتمكن فريق الدعم لدينا من مساعدتك بشكل أكبر."]
                },
                "end_glad_to_help_options": {"en": ["Glad I could help!"], "fr": ["Ravi d'aider!"], "ar": ["سعدت بالمساعدة!"]},
                "images_found_title_options": {"en": ["📷 Images for model {}:"], "fr": ["📷 Images pour le modèle {} :"], "ar": ["📷 صور للطراز {}:"]},
                "steps_batch_intro_first_options": {"en": ["Alright, let's try these initial steps:"], "fr": ["Bien, essayons ces premières étapes :"], "ar": ["حسنًا، لنجرب هذه الخطوات الأولية:"]},
                "steps_batch_intro_next_options": {"en": ["Okay, let's try the next set of steps:"], "fr": ["D'accord, essayons la série d'étapes suivante :"], "ar": ["تمام، لنجرب المجموعة التالية من الخطوات:"]},
                "step_connector_first_options": {"en": ["First,"], "fr": ["Premièrement,"], "ar": ["أولاً،"]},
                "step_connector_middle_options": {"en": ["Next,"], "fr": ["Ensuite,"], "ar": ["بعد ذلك،"]},
                "step_connector_final_options": {"en": ["Finally,"], "fr": ["Finalement,"], "ar": ["أخيرًا،"]},
                "resolved_prompt_options": {"en": ["Is the problem resolved?"], "fr": ["Le problème est-il résolu ?"], "ar": ["هل تم حل المشكلة؟"]}
            }
            
            for key, default_value_map in default_structure.items():
                if key not in self.prompts_db:
                    self.prompts_db[key] = default_value_map
                elif isinstance(default_value_map, dict): # Ensure all languages exist for this key
                    for lang_code, text_or_list in default_value_map.items():
                        if lang_code not in self.prompts_db[key]:
                            self.prompts_db[key][lang_code] = text_or_list
                        # If lang_code exists but is empty list (for _options keys)
                        elif isinstance(text_or_list, list) and \
                             isinstance(self.prompts_db[key].get(lang_code), list) and \
                             not self.prompts_db[key].get(lang_code):
                            self.prompts_db[key][lang_code] = text_or_list


        self.image_folder = Path(image_folder_path)
        self.static_image_url_prefix = static_image_url_prefix
        self.backend_base_url = backend_base_url.rstrip('/')

        try:
            self.nlp = NLPHelper()
        except Exception as e:
            raise RuntimeError(f"Could not initialize NLPHelper: {e}")

        self.translator = Translator()

    def _get_minimal_default_list_for_key(self, key, lang='en'):
        # These are ultimate fallbacks if prompt_db is corrupted or key truly missing
        defaults = {
            "escalate_to_support_intro_options": ["Please provide more details to our support team:"],
            "end_glad_to_help_options": ["Glad I could help!"],
            "images_found_title_options": ["Images for model {}:"],
            "steps_batch_intro_first_options": ["Alright, let's try these initial steps:"],
            "steps_batch_intro_next_options": ["Okay, let's try the next set of steps:"],
            "step_connector_first_options": ["First,"],
            "step_connector_middle_options": ["Next,"],
            "step_connector_final_options": ["Finally,"],
            "resolved_prompt_options": ["Is the problem resolved?"]
        }
        return defaults.get(key, ["Please proceed with the following:"])


    def _get_localized_list(self, key, lang='en'):
        lang_to_use = lang if lang in ['en', 'fr', 'ar'] else 'en'
        prompt_group_for_key = self.prompts_db.get(key)
        
        if not prompt_group_for_key or not isinstance(prompt_group_for_key, dict):
            return self._get_minimal_default_list_for_key(key, lang_to_use)

        specific_lang_list = prompt_group_for_key.get(lang_to_use)
        if isinstance(specific_lang_list, list) and specific_lang_list:
            return specific_lang_list
        
        english_list_default = prompt_group_for_key.get('en')
        if lang_to_use != 'en' and isinstance(english_list_default, list) and english_list_default:
            try:
                translated_list = [self._translate_text(item, lang_to_use, 'en') for item in english_list_default]
                # Check if translation produced different results and non-empty strings
                if any(translated_list[i] != english_list_default[i] and translated_list[i].strip() for i in range(min(len(translated_list), len(english_list_default)))):
                    return translated_list
            except Exception: pass
        
        if isinstance(english_list_default, list) and english_list_default:
             return english_list_default
        
        return self._get_minimal_default_list_for_key(key, lang_to_use)

    def _translate_text(self, text, dest_lang, src_lang='auto'):
        if not text or not text.strip(): return text 

        if src_lang != 'auto' and src_lang == dest_lang: return text
        
        try:
            actual_src_lang = src_lang
            if src_lang == 'auto':
                try:
                    detected = self.translator.detect(text)
                    detected_lang_code = detected.lang[0] if isinstance(detected.lang, list) else detected.lang
                    actual_src_lang = detected_lang_code.split('-')[0]
                    if actual_src_lang == dest_lang: return text
                except Exception: 
                    actual_src_lang = 'auto' 
            
            if actual_src_lang not in LANGUAGES and actual_src_lang != 'auto': actual_src_lang = 'auto'
            if dest_lang not in LANGUAGES: dest_lang = 'en'

            translated = self.translator.translate(text, dest=dest_lang, src=actual_src_lang)
            return translated.text
        except Exception as e:
            return text 

    def _detect_language(self, text):
        if not text or not text.strip(): return 'en'
        try:
            detected = self.translator.detect(text)
            lang_code = detected.lang[0] if isinstance(detected.lang, list) else detected.lang
            base_lang = lang_code.split('-')[0] 
            return base_lang if base_lang in ['en', 'fr', 'ar'] else 'en'
        except Exception as e:
            return 'en'

    def _get_localized_string(self, key, *args_tuple, lang='en'):
        lang_to_use = lang if lang in ['en', 'fr', 'ar'] else 'en'
        prompt_group = self.prompts_db.get(key, {}) 

        if not isinstance(prompt_group, dict): 
            list_options = self._get_localized_list(key, lang=lang_to_use) 
            base_text = list_options[0] if list_options else f"<Error: List key '{key}' empty or invalid>"
        else:
            default_english_text = prompt_group.get('en', f"<{key}_EN_NOT_FOUND>")
            base_text = default_english_text
            if lang_to_use in prompt_group:
                base_text = prompt_group[lang_to_use]
            elif lang_to_use != 'en' and default_english_text != f"<{key}_EN_NOT_FOUND>":
                base_text = self._translate_text(default_english_text, lang_to_use, 'en')
            
            if base_text == f"<{key}_EN_NOT_FOUND>" and not prompt_group.get('en'):
                 base_text = f"<Error: Prompt key '{key}' completely missing or EN text missing>"

        try:
            if "{}" in base_text and args_tuple:
                return base_text.format(*args_tuple)
            return base_text
        except Exception:
            unformatted_default_options = self.prompts_db.get(key, {}).get('en') if isinstance(self.prompts_db.get(key), dict) else self._get_localized_list(key, 'en')
            unformatted_default = unformatted_default_options if isinstance(unformatted_default_options, str) else (unformatted_default_options[0] if isinstance(unformatted_default_options, list) and unformatted_default_options else f"<Error formatting '{key}'>")
            return unformatted_default

    def extract_model_from_message(self, message_en):
        for intent in self.intents:
            for flow in intent.get("troubleshooting_flows", []):
                flow_pattern_str = flow.get("model_pattern", "")
                if not flow_pattern_str: continue
                individual_model_patterns = flow_pattern_str.split('|')
                for model_variant in individual_model_patterns:
                    model_variant_cleaned = model_variant.strip()
                    if not model_variant_cleaned: continue
                    pattern_re = r"(?:^|\b|\W)(" + re.escape(model_variant_cleaned).replace(r"\ ", r"[-\s]?") + r")(?:\b|\W|$)"
                    match = re.search(pattern_re, message_en, re.IGNORECASE)
                    if match: return match.group(1).strip(), flow
        return None, None

    def find_flow_by_model_name(self, model_name_input_en):
        std_model_name = model_name_input_en.strip().lower().replace('-', '').replace(' ', '')
        for intent in self.intents:
            for flow in intent.get("troubleshooting_flows", []):
                patterns = flow.get("model_pattern", "").split('|')
                for p_variant in patterns: 
                    std_p_variant = p_variant.strip().lower().replace('-', '').replace(' ', '')
                    if std_p_variant == std_model_name: return p_variant.strip(), flow
        return None, None

    def _extract_problem_after_model(self, message_en, model_name):
        model_name_pattern_part = re.escape(model_name).replace(r"\ ", r"[-\s]?")
        model_pattern_re = r"(?:^|\b|\W)" + model_name_pattern_part + r"(?:$|\b|\W)"
        desc_en = re.sub(model_pattern_re, " ", message_en, flags=re.IGNORECASE).strip()
        
        fillers = [
            r"^(my|the|a|an|it's|it is|i have|i'm having|there is|there's|it has|it says)\s+(problem|issue|trouble|error|code|message)\s+(with|on|for|regarding|about)\s*(my|the|a|an)?\s*",
            r"^(problem|issue|trouble|error|code|message)\s+(with|on|for|regarding|about)\s*(my|the|a|an)?\s*",
            r"^(is|has|having|shows|displays|got|comes up with|saying)\s+",
            r"\s+(tv|television|screen|display|set)\b", 
            r"^(tv|television|screen|display|set)\s+",
            r"^\s*(with|for|about|regarding|on)\s+", 
            r"^\s*(that|it)\s+", r"^\s*experiencing\s+",
            r"^\s*error\s*code\s*", r"^\s*code\s*", r"^\s*message\s*",
            r"\s+error\s*code\b", r"\s+code\b", r"\s+message\b",
            r"\.$", r"^\s*the\s+", r"^\s*a\s+",
        ]
        for pat in fillers: desc_en = re.sub(pat, "", desc_en, flags=re.IGNORECASE).strip()
        desc_en = re.sub(r'\s+',' ', desc_en).strip('.,;:!?"\'()[]{} ')
        
        if len(desc_en.split()) < 2 and desc_en.lower() in ["problem", "issue", "trouble", "error", "message"]: 
            return None
        return desc_en if desc_en and len(desc_en.split()) >= 1 else None

    def find_steps_for_problem(self, flow_details, problem_description_en):
        if not flow_details or "problems" not in flow_details: return None
        problems_data = flow_details["problems"]
        k_list, p_map = [], {}
        for p_data in problems_data:
            for kw in p_data.get("problem_keywords", []):
                k_list.append(kw)
                p_map[kw] = p_data["steps"]
        if not k_list or not problem_description_en: return None
        match_kw, sim = self.nlp.find_best_match(problem_description_en, k_list)
        return p_map[match_kw] if match_kw and sim >= 0.35 else None

    def _get_model_images_response(self, model_name, lang):
        if not model_name: return []
        safe_model = re.sub(r'[\\/*?:"<>|]', '_', model_name)
        pattern = re.escape(safe_model) + r"\.(?P<number>\d+)\.(jpg|jpeg|png|gif)$"
        img_info = []
        if not self.image_folder.is_dir(): return []
        for fname in os.listdir(self.image_folder):
            m = re.match(pattern, fname, re.IGNORECASE)
            if m:
                try: img_info.append((int(m.group("number")), fname))
                except ValueError: continue
        img_info.sort()
        if not img_info: return []
        
        gallery_items = []
        prefix = self.static_image_url_prefix.strip()
        if not prefix.startswith('/'): prefix = '/' + prefix
        if len(prefix) > 1: prefix = prefix.rstrip('/')

        for _, (num, filename) in enumerate(img_info):
            img_url = f"{self.backend_base_url}{prefix}/{filename}"
            alt_text_en = f"{model_name} illustration {num}" if model_name else f"Illustration {num}"
            alt = self._translate_text(alt_text_en, lang, 'en')
            gallery_items.append({"url": img_url, "alt": alt, "name": filename})
        if not gallery_items: return []
        
        title_options = self._get_localized_list("images_found_title_options", lang=lang)
        if not title_options: return []
        title_template = random.choice(title_options)
        try:
            title = title_template.format(model_name)
        except (IndexError, KeyError): 
            title = title_template.replace("{}", model_name or "device")
        return [{"type": "image_gallery", "text": title, "items": gallery_items}]

    def process_message(self, user_message_text, session_state, current_language, selected_mode):
        bot_responses = []
        user_lang_for_output = current_language 

        if session_state.get("bot_stage") == "ended":
            is_likely_new_problem = False
            if user_message_text and len(user_message_text.strip()) > 3:
                simple_end_confirmations = {"ok", "thanks", "thank you", "great", "cool", "bye", "goodbye", "👍", "ok.", "okay", "merci", "d'accord", "شكرا", "حسنا"}
                if user_message_text.lower().strip() not in simple_end_confirmations and len(user_message_text.split()) > 2:
                    is_likely_new_problem = True
                else:
                    translated_for_model_check = self._translate_text(user_message_text, 'en', self._detect_language(user_message_text))
                    temp_model_extract, _ = self.extract_model_from_message(translated_for_model_check)
                    if temp_model_extract: is_likely_new_problem = True
            
            if is_likely_new_problem: session_state.clear() 

        if "bot_stage" not in session_state:
            session_state.update({
                "bot_stage": "initial", "tv_model": None, "problem_description_en": None,
                "flow_for_model_obj": None, "current_actual_steps_en": None,
                "current_step_batch_start_index": 0, "is_first_batch_presentation": True
            })
        
        msg_en_for_processing = ""
        if user_message_text and user_message_text.strip():
            detected_input_lang = self._detect_language(user_message_text)
            if detected_input_lang != 'en':
                msg_en_for_processing = self._translate_text(user_message_text, 'en', src_lang=detected_input_lang)
            else: msg_en_for_processing = user_message_text
        else: msg_en_for_processing = user_message_text or ""


        def get_varied_prompt(key, lang_for_prompt, *args):
            options = self._get_localized_list(key, lang=lang_for_prompt)
            if not options: return f"<Error: No prompts for '{key}' in lang '{lang_for_prompt}'>"
            chosen_template = random.choice(options)
            try:
                return chosen_template.format(*args) if args and "{}" in chosen_template else chosen_template
            except (IndexError, KeyError):
                return chosen_template.replace("{}", (args[0] if args and args[0] is not None else "...") )

        # Function to add support form trigger
        def add_support_form_responses(current_model_en, current_problem_en):
            intro_text = get_varied_prompt("escalate_to_support_intro_options", user_lang_for_output)
            bot_responses.append({"type": "text", "content": intro_text, "sender":"bot", "timestamp":random.random()}) # Added sender/ts for consistency
            
            prefill_data = {}
            if current_model_en: prefill_data["model"] = current_model_en
            if current_problem_en: 
                problem_disp_for_form = self._translate_text(current_problem_en, user_lang_for_output, 'en') if user_lang_for_output != 'en' else current_problem_en
                prefill_data["problem_summary"] = problem_disp_for_form

            bot_responses.append({"type": "support_form", "prefill": prefill_data, "sender":"bot", "timestamp":random.random()})
            session_state["bot_stage"] = "ended_awaiting_support_submission" # New stage

        for _ in range(3): 
            current_bot_stage = session_state["bot_stage"]
            initial_stage_for_iteration = current_bot_stage
            
            if current_bot_stage == "initial":
                model_en, flow = self.extract_model_from_message(msg_en_for_processing)
                if model_en:
                    session_state.update({"tv_model": model_en, "flow_for_model_obj": flow})
                    problem_en = self._extract_problem_after_model(msg_en_for_processing, model_en)
                    if problem_en:
                        session_state["problem_description_en"] = problem_en
                        problem_disp = self._translate_text(problem_en, user_lang_for_output, 'en') if user_lang_for_output != 'en' else problem_en
                        bot_responses.append({"type": "text", "content": self._get_localized_string("model_problem_confirmation", model_en, problem_disp, lang=user_lang_for_output)})
                        session_state["bot_stage"] = "ready_for_steps"
                    else:
                        bot_responses.append({"type": "text", "content": self._get_localized_string("model_known_ask_problem", model_en, lang=user_lang_for_output)})
                        session_state["bot_stage"] = "awaiting_problem"
                    img_resps = self._get_model_images_response(model_en, user_lang_for_output)
                    if img_resps: bot_responses.extend(img_resps)
                else:
                    if msg_en_for_processing and len(msg_en_for_processing.split()) > 1:
                        session_state["problem_description_en"] = msg_en_for_processing
                        problem_disp = self._translate_text(msg_en_for_processing, user_lang_for_output, 'en') if user_lang_for_output != 'en' else msg_en_for_processing
                        bot_responses.append({"type": "text", "content": self._get_localized_string("problem_known_ask_model", problem_disp, lang=user_lang_for_output)})
                    else:
                        bot_responses.append({"type": "text", "content": self._get_localized_string("prompt_model_not_detected", lang=user_lang_for_output)})
                    session_state["bot_stage"] = "awaiting_model"

            elif current_bot_stage == "awaiting_model":
                model_en, flow = self.find_flow_by_model_name(msg_en_for_processing)
                if model_en:
                    session_state.update({"tv_model": model_en, "flow_for_model_obj": flow})
                    problem_desc_en_from_session = session_state.get("problem_description_en")
                    if problem_desc_en_from_session:
                        problem_disp = self._translate_text(problem_desc_en_from_session, user_lang_for_output, 'en') if user_lang_for_output != 'en' else problem_desc_en_from_session
                        bot_responses.append({"type": "text", "content": self._get_localized_string("model_problem_confirmation", model_en, problem_disp, lang=user_lang_for_output)})
                        session_state["bot_stage"] = "ready_for_steps"
                    else:
                        bot_responses.append({"type": "text", "content": self._get_localized_string("model_known_ask_problem", model_en, lang=user_lang_for_output)})
                        session_state["bot_stage"] = "awaiting_problem"
                    img_resps = self._get_model_images_response(model_en, user_lang_for_output)
                    if img_resps: bot_responses.extend(img_resps)
                else: 
                    bot_responses.append({"type": "text", "content": self._get_localized_string("model_not_in_db", lang=user_lang_for_output)})
                    problem_desc_val = session_state.get("problem_description_en")
                    if problem_desc_val:
                        problem_disp = self._translate_text(problem_desc_val, user_lang_for_output, 'en') if user_lang_for_output != 'en' else problem_desc_val
                        bot_responses.append({"type": "text", "content": self._get_localized_string("problem_known_ask_model", problem_disp, lang=user_lang_for_output)})
                    else:
                        bot_responses.append({"type": "text", "content": self._get_localized_string("prompt_model_not_detected", lang=user_lang_for_output)})

            elif current_bot_stage == "awaiting_problem":
                model_en = session_state.get("tv_model")
                if not model_en: 
                    bot_responses.append({"type": "text", "content": self._get_localized_string("prompt_model_not_detected", lang=user_lang_for_output)})
                    session_state["bot_stage"] = "awaiting_model"; break 
                
                problem_en_from_input = msg_en_for_processing
                if len(problem_en_from_input.split()) < 2 :
                     problem_en_from_input = self._extract_problem_after_model(msg_en_for_processing, model_en) or problem_en_from_input

                session_state["problem_description_en"] = problem_en_from_input
                problem_disp = self._translate_text(problem_en_from_input, user_lang_for_output, 'en') if user_lang_for_output != 'en' else problem_en_from_input
                bot_responses.append({"type": "text", "content": self._get_localized_string("model_problem_confirmation", model_en, problem_disp, lang=user_lang_for_output)})
                session_state["bot_stage"] = "ready_for_steps"
            
            if session_state.get("bot_stage") == "ready_for_steps": 
                model_en, problem_en = session_state.get("tv_model"), session_state.get("problem_description_en")
                flow = session_state.get("flow_for_model_obj")

                if not (model_en and problem_en): 
                    bot_responses.append({"type": "text", "content": self._get_localized_string("fallback_understand", lang=user_lang_for_output)})
                    session_state["bot_stage"] = "initial"; break 
                if not flow:
                    add_support_form_responses(model_en, problem_en) # No flow for model
                    break 
                
                full_steps_en = self.find_steps_for_problem(flow, problem_en) 
                if full_steps_en and len(full_steps_en) > 0:
                    actual_steps = full_steps_en[1:] if len(full_steps_en) > 1 else [] 
                    session_state["current_actual_steps_en"] = actual_steps
                    if actual_steps:
                        session_state.update({"current_step_batch_start_index": 0, "is_first_batch_presentation": True, "bot_stage": "presenting_step_batch"})
                    else: 
                        bot_responses.append({"type": "text", "content": get_varied_prompt("resolved_prompt_options", user_lang_for_output)})
                        session_state["bot_stage"] = "awaiting_resolution_feedback"
                else: 
                    add_support_form_responses(model_en, problem_en) # No steps for problem
                current_bot_stage = session_state["bot_stage"]

            if current_bot_stage == "presenting_step_batch":
                actual_steps = session_state.get("current_actual_steps_en", [])
                batch_start_idx = session_state.get("current_step_batch_start_index", 0)
                if not actual_steps or batch_start_idx >= len(actual_steps):
                    add_support_form_responses(session_state.get("tv_model"), session_state.get("problem_description_en")) # All steps exhausted
                    session_state["bot_stage"] = "ended_awaiting_support_submission"
                else:
                    batch_en = actual_steps[batch_start_idx : batch_start_idx + self.MAX_STEPS_PER_BATCH]
                    intro_key = "steps_batch_intro_first_options" if session_state.get("is_first_batch_presentation", True) else "steps_batch_intro_next_options"
                    session_state["is_first_batch_presentation"] = False
                    lead_in = get_varied_prompt(intro_key, user_lang_for_output)
                    
                    proc_steps_loc = []
                    for step_en_item in batch_en: 
                        step_loc = self._translate_text(step_en_item, user_lang_for_output, 'en') 
                        step_clean = re.sub(r"^\s*([\d\w][.)]\s*)+", "", step_loc.strip()).strip()
                        if not step_clean: continue
                        if not re.search(r'[.!?]$', step_clean): step_clean += "."
                        proc_steps_loc.append(step_clean[0].upper() + step_clean[1:])

                    if not proc_steps_loc:
                         add_support_form_responses(session_state.get("tv_model"), session_state.get("problem_description_en")) # No valid steps in batch
                         session_state["bot_stage"] = "ended_awaiting_support_submission"
                    else:
                        para_parts = []
                        for i, step_txt in enumerate(proc_steps_loc):
                            conn = ""
                            if len(proc_steps_loc) > 1:
                                conn_key = "step_connector_first_options" if i == 0 else ("step_connector_final_options" if i == len(proc_steps_loc) - 1 else "step_connector_middle_options")
                                conn = get_varied_prompt(conn_key, user_lang_for_output) + " "
                            para_parts.append(conn + step_txt)
                        
                        full_steps_txt = " ".join(para_parts)
                        final_lead_in = lead_in.strip()
                        if final_lead_in and not final_lead_in.endswith(' '): final_lead_in += " "
                        bot_responses.append({"type": "text", "content": f"{final_lead_in}{full_steps_txt}"})
                        bot_responses.append({"type": "text", "content": get_varied_prompt("resolved_prompt_options", user_lang_for_output)})
                        session_state["current_step_batch_start_index"] = batch_start_idx + len(batch_en)
                        session_state["bot_stage"] = "awaiting_resolution_feedback"
                current_bot_stage = session_state["bot_stage"] 

            elif current_bot_stage == "awaiting_resolution_feedback":
                feedback_en_lower = msg_en_for_processing.lower() 
                pos_s = {"perfect", "excellent", "solved", "all fixed", "completely fixed", "definitely yes", "absolutely", "all good", "works now", "that's it", "totally fixed"}
                pos_g = {"yes", "y", "yeah", "ok", "okay", "sure", "affirmative", "resolved", "fixed", "correct", "good", "fine", "got it", "it worked", "success", "oui", "نعم", "si", "ja"}
                neg_s = {"no way", "not at all", "definitely not", "still broken", "completely broken", "failed miserably", "zero change", "nothing happened"}
                neg_g = {"no", "n", "nope", "negative", "not ok", "not resolved", "not fixed", "still", "doesn't work", "didn't work", "problem persists", "issue remains", "failed", "no change", "same problem", "issue is still there", "non", "لا", "nein"}
                inc_kws = {"almost", "partially", "a little", "a bit", "not quite", "not really", "not completely", "somewhat", "kinda", "sorta", "still some issues", "better but", "improved but"}

                said_inc = any(re.search(r'\b' + kw + r'\b', feedback_en_lower) for kw in inc_kws)
                said_neg = any(re.search(r'\b' + kw + r'\b', feedback_en_lower) for kw in neg_s.union(neg_g))
                said_pos_strong = any(re.search(r'\b' + kw + r'\b', feedback_en_lower) for kw in pos_s)
                said_pos_general = any(re.search(r'\b' + kw + r'\b', feedback_en_lower) for kw in pos_g)
                
                user_sentiment = "fallback"
                if said_neg or said_inc: user_sentiment = "negative"
                if (said_pos_strong or said_pos_general) and not (said_neg or said_inc): user_sentiment = "positive"
                elif said_pos_general and said_neg: user_sentiment = "fallback"

                if user_sentiment == "positive":
                    bot_responses.append({"type": "text", "content": get_varied_prompt("end_glad_to_help_options", user_lang_for_output)})
                    session_state["bot_stage"] = "ended"
                elif user_sentiment == "negative":
                    actual_steps, next_idx = session_state.get("current_actual_steps_en", []), session_state.get("current_step_batch_start_index", 0)
                    if not actual_steps or next_idx >= len(actual_steps): 
                        add_support_form_responses(session_state.get("tv_model"), session_state.get("problem_description_en")) # All steps exhausted after negative feedback
                    else: 
                        session_state["bot_stage"] = "presenting_step_batch" 
                else: 
                    bot_responses.append({"type": "text", "content": self._get_localized_string("fallback_understand", lang=user_lang_for_output)})
                    bot_responses.append({"type": "text", "content": get_varied_prompt("resolved_prompt_options", user_lang_for_output)}) 
                current_bot_stage = session_state["bot_stage"]
            
            # If stage is now one that signals end or waiting for form, break processing loop
            if current_bot_stage in ["ended", "ended_awaiting_support_submission"]:
                break

            if initial_stage_for_iteration == current_bot_stage and current_bot_stage not in ["initial", "awaiting_model", "awaiting_problem", "awaiting_resolution_feedback"]:
                break 
            if current_bot_stage in ["awaiting_model", "awaiting_problem", "awaiting_resolution_feedback", "initial"]: # Also break for "initial" if it didn't progress
                 break


        final_bot_stage = session_state.get("bot_stage")
        if not bot_responses and user_message_text and user_message_text.strip() and final_bot_stage not in ["ended", "ended_awaiting_support_submission"]:
            bot_responses.append({"type": "text", "content": self._get_localized_string("fallback_understand", lang=user_lang_for_output)})
        
        # If ended normally (not via support form trigger) and no specific end message, add one.
        if final_bot_stage == "ended":
            is_any_text_response = any(r.get("type") == "text" and r.get("content") for r in bot_responses)
            if not is_any_text_response and (user_message_text and user_message_text.strip()):
                 bot_responses.append({"type": "text", "content": get_varied_prompt("end_glad_to_help_options", user_lang_for_output)})
            elif not bot_responses : # if truly no response at all and ended
                 bot_responses.append({"type": "text", "content": get_varied_prompt("end_glad_to_help_options", user_lang_for_output)})


        return bot_responses
