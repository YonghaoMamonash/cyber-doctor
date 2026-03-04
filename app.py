import base64
from qa.answer import get_answer
from qa.question_parser import parse_question
from qa.function_tool import process_image_describe_tool
from qa.purpose_type import userPurposeType
from audio.audio_generate import audio_generate

import PyPDF2
import chardet
import mimetypes
import gradio as gr
from icecream import ic
from docx import Document
from pydub import AudioSegment
import speech_recognition as sr
from opencc import OpenCC
import os

from utils.answer_guard import ANSWER_FALLBACK_MESSAGE, is_valid_answer_payload
from utils.chat_ui import format_ai_response_as_cards
from utils.theme_mode import THEME_MODE_OPTIONS, THEME_MODE_SYSTEM, normalize_theme_mode


AVATAR = ("resource/user.png", "resource/bot.jpg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MEDICAL_THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@500;600;700&family=Noto+Sans:wght@400;500;700&display=swap');

:root {
    --healing-blue: #2563EB;
    --calm-blue: #1E3A8A;
    --deep-olive: #3F6212;
    --bg-soft: #F9FAFB;
    --surface: #FFFFFF;
    --surface-soft: #F8FBFF;
    --border-soft: #D8E1F0;
    --card-border: #DBE5F6;
    --text-main: #0F172A;
    --text-muted: #6B7280;
    --shadow-soft: 0 12px 28px rgba(15, 23, 42, 0.10);
    --shadow-strong: 0 20px 45px rgba(30, 58, 138, 0.16);
    --theme-chip-bg: #FFFFFF;
    --theme-chip-border: rgba(37, 99, 235, 0.28);
    --theme-chip-text: #1E3A8A;
    --theme-chip-active-bg: #2563EB;
    --theme-chip-active-text: #FFFFFF;
}

[data-theme="dark"] {
    --healing-blue: #60A5FA;
    --calm-blue: #BFDBFE;
    --deep-olive: #A3E635;
    --bg-soft: #0B1220;
    --surface: #111827;
    --surface-soft: #1A2437;
    --border-soft: #334155;
    --card-border: #334155;
    --text-main: #E5E7EB;
    --text-muted: #94A3B8;
    --shadow-soft: 0 12px 30px rgba(2, 6, 23, 0.52);
    --shadow-strong: 0 24px 50px rgba(2, 6, 23, 0.65);
    --theme-chip-bg: #111827;
    --theme-chip-border: #334155;
    --theme-chip-text: #CBD5E1;
    --theme-chip-active-bg: #1D4ED8;
    --theme-chip-active-text: #EFF6FF;
}

.gradio-container {
    font-family: 'Noto Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    color: var(--text-main);
    background:
        radial-gradient(circle at 92% 4%, rgba(37, 99, 235, 0.14), transparent 40%),
        var(--bg-soft);
    padding-bottom: 84px;
    transition: background-color 180ms ease, color 180ms ease;
}

[data-theme="dark"] .gradio-container {
    background:
        radial-gradient(circle at 92% 4%, rgba(59, 130, 246, 0.30), transparent 42%),
        linear-gradient(180deg, #0B1220 0%, #0F172A 100%);
}

#app-title h1 {
    font-family: 'Figtree', 'Noto Sans', sans-serif;
    color: var(--calm-blue);
    letter-spacing: 0.02em;
    font-weight: 700;
}

#theme-toggle-row {
    margin-top: -4px;
    margin-bottom: 10px;
}

#theme-mode .wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

#theme-mode .wrap label {
    border-radius: 999px !important;
    border: 1px solid var(--theme-chip-border) !important;
    background: var(--theme-chip-bg) !important;
    color: var(--theme-chip-text) !important;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
    transition: all 180ms ease;
    cursor: pointer !important;
    padding: 0 10px !important;
}

#theme-mode .wrap label:hover {
    box-shadow: 0 12px 26px rgba(15, 23, 42, 0.18);
    transform: translateY(-1px);
}

#theme-mode .wrap label:has(input:checked),
#theme-mode .wrap label.selected,
#theme-mode .wrap label[data-selected="true"] {
    background: var(--theme-chip-active-bg) !important;
    border-color: var(--theme-chip-active-bg) !important;
    color: var(--theme-chip-active-text) !important;
}

#cyber-chatbot {
    border-radius: 20px;
    border: 1px solid var(--border-soft);
    background: linear-gradient(180deg, var(--surface) 0%, var(--surface-soft) 100%);
    box-shadow: var(--shadow-strong);
}

#cyber-chatbot .message {
    border-radius: 16px !important;
    border: 1px solid var(--card-border) !important;
    background: var(--surface) !important;
    color: var(--text-main) !important;
    box-shadow: var(--shadow-soft);
}

#cyber-chatbot .message.user {
    border-left: 4px solid var(--healing-blue) !important;
}

#cyber-chatbot .message.bot {
    border-left: 4px solid var(--deep-olive) !important;
}

#guide-row {
    gap: 8px;
    flex-wrap: wrap;
}

#guide-row .guide-pill {
    border-radius: 999px !important;
    border: 1px solid rgba(37, 99, 235, 0.42) !important;
    background: var(--surface) !important;
    color: var(--calm-blue) !important;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.12);
    transition: all 180ms ease;
    cursor: pointer !important;
}

#guide-row .guide-pill:hover {
    border-color: var(--calm-blue) !important;
    box-shadow: 0 12px 26px rgba(30, 58, 138, 0.20);
    transform: translateY(-1px);
}

#cyber-input {
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    background: var(--surface);
    box-shadow: var(--shadow-soft);
}

#cyber-input textarea,
#cyber-input input {
    color: var(--text-main) !important;
}

.cyber-result-grid {
    display: grid;
    gap: 10px;
}

.cyber-result-card {
    background: var(--surface);
    border-radius: 14px;
    border: 1px solid var(--card-border);
    box-shadow: var(--shadow-soft);
    padding: 12px 14px;
    animation: card-fade 220ms ease-out both;
}

.cyber-result-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--calm-blue);
    font-weight: 700;
    margin-bottom: 6px;
}

.cyber-card-icon {
    font-size: 18px;
    line-height: 1;
}

.cyber-result-card-body {
    color: var(--text-main);
    line-height: 1.62;
    font-size: 14px;
}

#medical-disclaimer {
    position: fixed;
    left: 50%;
    bottom: 14px;
    transform: translateX(-50%);
    width: min(920px, calc(100% - 24px));
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.80);
    border: 1px solid rgba(148, 163, 184, 0.52);
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
    backdrop-filter: blur(8px);
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
    padding: 8px 14px;
    z-index: 1000;
}

[data-theme="dark"] #medical-disclaimer {
    background: rgba(15, 23, 42, 0.76);
    border: 1px solid rgba(71, 85, 105, 0.80);
}

@keyframes card-fade {
    from {
        opacity: 0;
        transform: translateY(5px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 768px) {
    #theme-mode .wrap label {
        min-height: 36px;
        font-size: 13px !important;
    }

    #guide-row .guide-pill {
        min-height: 38px;
        font-size: 13px !important;
    }

    #medical-disclaimer {
        width: calc(100% - 16px);
        bottom: 8px;
        border-radius: 12px;
    }
}

@media (prefers-reduced-motion: reduce) {
    .cyber-result-card,
    #guide-row .guide-pill {
        animation: none !important;
        transition: none !important;
    }
}
"""

MEDICAL_DISCLAIMER_HTML = """
<div id="medical-disclaimer">
医学免责声明：本系统仅用于健康教育与参考，不替代执业医师面对面诊断、处方或急救处理。
</div>
"""

THEME_BOOTSTRAP_JS = """
(currentMode) => {
    const validModes = ["浅色", "深色", "跟随系统"];
    const modeLabelToKey = { "浅色": "light", "深色": "dark", "跟随系统": "system" };

    const setupThemeController = () => {
        if (window.cyberDoctorThemeController) {
            return window.cyberDoctorThemeController;
        }

        const media = window.matchMedia("(prefers-color-scheme: dark)");

        const apply = (modeLabel) => {
            const safeMode = validModes.includes(modeLabel) ? modeLabel : "跟随系统";
            const requested = modeLabelToKey[safeMode];
            const effectiveTheme = requested === "system" ? (media.matches ? "dark" : "light") : requested;

            document.documentElement.setAttribute("data-theme", effectiveTheme);
            document.documentElement.style.colorScheme = effectiveTheme;
            localStorage.setItem("cyber_doctor_theme_mode", safeMode);
            return safeMode;
        };

        const onSystemChange = () => {
            const savedMode = localStorage.getItem("cyber_doctor_theme_mode") || "跟随系统";
            if (savedMode === "跟随系统") {
                apply("跟随系统");
            }
        };

        if (!window.__cyberDoctorThemeMediaBound) {
            if (typeof media.addEventListener === "function") {
                media.addEventListener("change", onSystemChange);
            } else if (typeof media.addListener === "function") {
                media.addListener(onSystemChange);
            }
            window.__cyberDoctorThemeMediaBound = true;
        }

        window.cyberDoctorThemeController = { apply };
        return window.cyberDoctorThemeController;
    };

    const controller = setupThemeController();
    const savedMode = localStorage.getItem("cyber_doctor_theme_mode");
    const initialMode = validModes.includes(savedMode)
        ? savedMode
        : (validModes.includes(currentMode) ? currentMode : "跟随系统");
    return controller.apply(initialMode);
}
"""

THEME_CHANGE_JS = """
(mode) => {
    const validModes = ["浅色", "深色", "跟随系统"];
    if (!window.cyberDoctorThemeController || typeof window.cyberDoctorThemeController.apply !== "function") {
        return validModes.includes(mode) ? mode : "跟随系统";
    }
    return window.cyberDoctorThemeController.apply(mode);
}
"""

QUICK_GUIDE_PROMPTS = [
    ("症状初筛", "请根据我的症状做分级判断，并告诉我是否需要立即就医。"),
    ("用药提醒", "请给我常见安全用药建议，并标注禁忌与注意事项。"),
    ("风险预警", "请帮我识别当前情况的风险点，并按高-中-低风险列出。"),
    ("饮食调理", "请根据常见慢病管理给我一份可执行的饮食建议。"),
    ("复诊建议", "请给出我何时复诊、需要携带哪些检查资料的建议。"),
]

# pip install whisper
# pip install openai-whisper
# pip install soundfile
# pip install pydub
# pip install opencc-python-reimplemented


def convert_to_simplified(text):
    converter = OpenCC("t2s")
    return converter.convert(text)


def convert_audio_to_wav(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)  # 自动识别格式
    wav_file_path = audio_file_path.rsplit(".", 1)[0] + ".wav"  # 生成 WAV 文件路径
    audio.export(wav_file_path, format="wav")  # 将音频文件导出为 WAV 格式
    return wav_file_path


def audio_to_text(audio_file_path):
    # 创建识别器对象
    # 如果不是 WAV 格式，先转换为 WAV
    if not audio_file_path.endswith(".wav"):
        audio_file_path = convert_audio_to_wav(audio_file_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        # 使用 Google Web Speech API 进行语音识别，不用下载模型但对网络要求高
        # text = recognizer.recognize_google(audio_data, language="zh-CN")
        # 使用 whisper 进行语音识别，自动下载模型到本地
        text = recognizer.recognize_whisper(audio_data, language="zh")
        text_simplified = convert_to_simplified(text)
    return text_simplified


# pip install PyPDF2
def pdf_to_str(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def docx_to_str(file_path):
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)


# pip install chardet
def text_file_to_str(text_file):
    with open(text_file, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]

    # 使用检测到的编码来读取文件
    with open(text_file, "r", encoding=encoding) as file:
        return file.read()


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string


def render_response_cards(message: str) -> str:
    card_markup = format_ai_response_as_cards(message)
    if card_markup:
        return card_markup
    return message


def submit_quick_prompt(chatbot, prompt):
    chat_input = {"text": prompt, "files": []}
    yield from grodio_view(chatbot, chat_input)


def build_quick_prompt_handler(prompt):
    def _handler(chatbot):
        yield from submit_quick_prompt(chatbot, prompt)

    return _handler


# 核心函数
def grodio_view(chatbot, chat_input):

    # 用户消息立即显示
    user_message = chat_input["text"]
    bot_response = "loading..."
    chatbot.append([user_message, bot_response])
    yield chatbot

    # 处理用户上传的文件
    files = chat_input["files"]
    audios = []
    images = []
    pdfs = []
    docxs = []
    texts = []

    for file in files:
        file_type, _ = mimetypes.guess_type(file)
        if file_type.startswith("audio/"):
            audios.append(file)
        elif file_type.startswith("image/"):
            images.append(file)
        elif file_type.startswith("application/pdf"):
            pdfs.append(file)
        elif file_type.startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            docxs.append(file)
        elif file_type.startswith("text/"):
            texts.append(file)
        else:
            user_message += "请你将下面的句子修饰后输出，不要包含额外的文字，句子:'该文件为不支持的文件类型'"
            print(f"Unknown file type: {file_type}")

    # 图片文件解析
    if images != []:
        image_url = images
        image_base64 = [image_to_base64(image) for image in image_url]

        for i, image in enumerate(image_base64):
            chatbot[-1][
                0
            ] += f"""
                <div>
                    <img src="data:image/png;base64,{image}" alt="Generated Image" style="max-width: 100%; height: auto; cursor: pointer;" />
                </div>
                """
            yield chatbot
    else:
        image_url = None

    question_type = parse_question(user_message, image_url)
    ic(question_type)

    # 音频文件解析
    if audios != []:
        for i, audio in enumerate(audios):
            audio_message = audio_to_text(audio)
            if audio_message == "":
                user_message += "请你将下面的句子修饰后输出，不要包含额外的文字，句子:'音频识别失败，请稍后再试'"
            elif "作曲" in audio_message:
                user_message += "请你将下面的句子修饰后输出，不要包含额外的文字，句子:'不好意思，我无法理解音乐'"
            else:
                user_message += f"音频{i+1}内容：{audio_message}"

    if pdfs != []:
        for i, pdf in enumerate(pdfs):
            pdf_text = pdf_to_str(pdf)
            user_message += f"PDF{i+1}内容：{pdf_text}"

    if docxs != []:
        for i, docx in enumerate(docxs):
            docx_text = docx_to_str(docx)
            user_message += f"DOCX{i+1}内容：{docx_text}"

    if texts != []:
        for i, text in enumerate(texts):
            text_string = text_file_to_str(text)
            user_message += f"文本{i+1}内容：{text_string}"

    if user_message == "":
        user_message = "请你将下面的句子修饰后输出，不要包含额外的文字，句子:'请问您有什么想了解的，我将尽力为您服务'"
    answer = get_answer(user_message, chatbot, question_type, image_url)
    if not is_valid_answer_payload(answer):
        chatbot[-1][1] = ANSWER_FALLBACK_MESSAGE
        yield chatbot
        return
    bot_response = ""

    # 处理文本生成/其他/文档检索/知识图谱检索
    if (
        answer[1] == userPurposeType.text
        or answer[1] == userPurposeType.RAG
        or answer[1] == userPurposeType.KnowledgeGraph
    ):
        # 流式输出
        for chunk in answer[0]:
            bot_response = bot_response + (chunk.choices[0].delta.content or "")
            chatbot[-1][1] = bot_response
            yield chatbot
        chatbot[-1][1] = render_response_cards(bot_response)
        yield chatbot

    # 处理图片生成
    if answer[1] == userPurposeType.ImageGeneration:
        image_url = answer[0]
        describe = process_image_describe_tool(
            question_type=userPurposeType.ImageDescribe,
            question="描述这个图片，不要识别‘AI生成’",
            history="",
            image_url=[image_url],
        )
        combined_message = f"""
            **生成的图片:**
            ![Generated Image]({image_url})
            {describe[0]}
            """
        chatbot[-1][1] = combined_message
        yield chatbot

    # 处理图片描述
    if answer[1] == userPurposeType.ImageDescribe:
        for i in range(0, len(answer[0]), 1):
            bot_response += answer[0][i : i + 1]  # 累加当前chunk到combined_message
            chatbot[-1][1] = bot_response  # 更新chatbot对话中的最后一条消息
            yield chatbot  # 实时输出当前累积的对话内容
        chatbot[-1][1] = render_response_cards(bot_response)
        yield chatbot

    # 处理视频
    if answer[1] == userPurposeType.Video:
        if answer[0] is not None:
            chatbot[-1][1] = answer[0]
        else:
            chatbot[-1][1] = "抱歉，视频生成失败，请稍后再试"
        yield chatbot

    # 处理PPT
    if answer[1] == userPurposeType.PPT:
        if answer[0] is not None:
            chatbot[-1][1] = answer[0]
        else:
            chatbot[-1][1] = "抱歉，PPT生成失败，请稍后再试"
        yield chatbot

    # 处理Docx
    if answer[1] == userPurposeType.Docx:
        if answer[0] is not None:
            chatbot[-1][1] = answer[0]
        else:
            chatbot[-1][1] = "抱歉，文档生成失败，请稍后再试"
        yield chatbot

    # 处理音频生成
    if answer[1] == userPurposeType.Audio:
        if answer[0] is not None:
            chatbot[-1][1] = answer[0]
        else:
            chatbot[-1][1] = "抱歉，音频生成失败，请稍后再试"
        yield chatbot

    # 处理联网搜索
    if answer[1] == userPurposeType.InternetSearch:
        if answer[3] == False:
            reference_message = (
                "由于网络问题，访问互联网失败，下面由我根据现有知识给出回答："
            )
        else:
            # 将字典中的内容转换为 Markdown 格式的链接
            links = "\n".join(f"[{title}]({link})" for link, title in answer[2].items())
            links += "\n"
            reference_message = f"参考资料：{links}"
        for i in range(0, len(reference_message)):
            bot_response = reference_message[: i + 1]
            chatbot[-1][1] = bot_response
            yield chatbot
        internet_answer = ""
        for chunk in answer[0]:
            internet_answer = internet_answer + (chunk.choices[0].delta.content or "")
            bot_response = reference_message + internet_answer
            chatbot[-1][1] = bot_response
            yield chatbot
        chatbot[-1][1] = (
            f"{reference_message}\n{render_response_cards(internet_answer)}"
            if internet_answer
            else reference_message
        )
        yield chatbot


def gradio_audio_view(chatbot, audio_input):

    # 用户消息立即显示
    if audio_input is None:
        user_message = ""
    else:
        user_message = (audio_input, "audio")
    bot_response = "loading..."
    chatbot.append([user_message, bot_response])
    yield chatbot

    if audio_input is None:
        audio_message = "无音频"
    else:
        audio_message = audio_to_text(audio_input)

    chatbot[-1][0] = audio_message

    user_message = ""
    if audio_message == "无音频":
        user_message += "请你将下面的句子修饰后输出，不要包含额外的文字，句子:'欢迎与我对话，我将用语音回答您'"
    elif audio_message == "":
        user_message += "请你将下面的句子修饰后输出，不要包含额外的文字，句子:'音频识别失败，请稍后再试'"
    elif "作曲 作曲" in audio_message:
        user_message += "请你将下面的句子修饰后输出，不要包含额外的文字，句子:'不好意思，我无法理解音乐'"
    else:
        user_message += audio_message

    if user_message == "":
        user_message = "请你将下面的句子修饰后输出，不要包含额外的文字，句子:'请问您有什么想了解的，我将尽力为您服务'"

    question_type = parse_question(user_message)
    ic(question_type)
    answer = get_answer(user_message, chatbot, question_type)
    if not is_valid_answer_payload(answer):
        chatbot[-1][1] = ANSWER_FALLBACK_MESSAGE
        yield chatbot
        return
    bot_response = ""

    # 处理文本生成/其他/文档检索/知识图谱检索
    if (
        answer[1] == userPurposeType.text
        or answer[1] == userPurposeType.RAG
        or answer[1] == userPurposeType.KnowledgeGraph
    ):
        # 语音输出
        for chunk in answer[0]:
            # 获取每个块的数据
            chunk_content = chunk.choices[0].delta.content or ""
            bot_response += chunk_content

        try:
            chatbot[-1][1] = (
                audio_generate(
                    text=bot_response,
                    model_name="zh-CN-YunxiNeural",
                ),
                "audio",
            )
        except Exception as e:
            print(f"音频生成失败，直接返回文本: {str(e)}")
            chatbot[-1][1] = bot_response 
            
        yield chatbot

    # 处理图片生成
    if answer[1] == userPurposeType.ImageGeneration:
        image_url = answer[0]
        describe = process_image_describe_tool(
            question_type=userPurposeType.ImageDescribe,
            question="描述这个图片，不要识别‘AI生成’",
            history=" ",
            image_url=[image_url],
        )
        combined_message = f"""
            **生成的图片:**
            ![Generated Image]({image_url})
            {describe[0]}
            """
        chatbot[-1][1] = combined_message
        yield chatbot

    # 处理视频
    if answer[1] == userPurposeType.Video:
        if answer[0] is not None:
            chatbot[-1][1] = answer[0]
        else:
            try:
                chatbot[-1][1] = (
                    audio_generate(
                        text="抱歉，视频生成失败，请稍后再试",
                        model_name="zh-CN-YunxiNeural",
                    ),
                    "audio",
                )
            except Exception as e:
                chatbot[-1][1] = "抱歉，视频生成失败，请稍后再试"
        yield chatbot

    # 处理PPT
    if answer[1] == userPurposeType.PPT:
        if answer[0] is not None:
            chatbot[-1][1] = answer[0]
        else:
            try:
                chatbot[-1][1] = (
                    audio_generate(
                        text="抱歉，PPT生成失败，请稍后再试",
                        model_name="zh-CN-YunxiNeural",
                    ),
                    "audio",
                )
            except Exception as e:
                chatbot[-1][1] = "抱歉，PPT生成失败，请稍后再试"
        yield chatbot

    # 处理Docx
    if answer[1] == userPurposeType.Docx:
        if answer[0] is not None:
            chatbot[-1][1] = answer[0]
        else:
            try:
                chatbot[-1][1] = (
                    audio_generate(
                        text="抱歉，文档生成失败，请稍后再试",
                        model_name="zh-CN-YunxiNeural",
                    ),
                    "audio",
                )
            except Exception as e:
                chatbot[-1][1] = "抱歉，文档生成失败，请稍后再试"
        yield chatbot

    # 处理音频生成
    if answer[1] == userPurposeType.Audio:
        if answer[0] is not None:
            chatbot[-1][1] = answer[0]
        else:
            try:
                chatbot[-1][1] = (
                    audio_generate(
                        text="抱歉，音频生成失败，请稍后再试",
                        model_name="zh-CN-YunxiNeural",
                    ),
                    "audio",
                )
            except Exception as e:
                chatbot[-1][1] = "抱歉，音频生成失败，请稍后再试"
        yield chatbot

    # 处理联网搜索
    if answer[1] == userPurposeType.InternetSearch:
        if answer[3] == False:
            bot_response = (
                "由于网络问题，访问互联网失败，下面由我根据现有知识给出回答："
            )
        # 语音输出
        for chunk in answer[0]:
            # 获取每个块的数据
            chunk_content = chunk.choices[0].delta.content or ""
            bot_response += chunk_content

        try:
            chatbot[-1][1] = (
                audio_generate(
                    text=bot_response,
                    model_name="zh-CN-YunxiNeural",
                ),
                "audio",
            )
        except Exception as e:
            print(f"音频生成失败，直接返回文本: {str(e)}")
            chatbot[-1][1] = bot_response
        yield chatbot


# 切换到语音模式的函数
def toggle_voice_mode():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
    )


# 切换回文本模式的函数
def toggle_text_mode():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


examples = [
    {"text": "您好", "files": []},
    {"text": "糖尿病的常见症状有哪些？", "files": []},
    {"text": "用语音重新回答我一次", "files": []},
    {"text": "帮我搜索一下养生知识", "files": []},
        {"text": "帮我生成一张老人练太极图片", "files": []},
    {
        "text": "帮我生成一份用于科普糖尿病发病原因，症状，治疗药物，预防措施的PPT",
        "files": [],
    },
    {"text": "请根据我给的参考资料，给我一个合理的饮食建议", "files": []},
    {"text": "请根据我给的参考资料，生成一个用于科普合理膳食的word", "files": []},
    {"text": "我最近想打太极养生，帮我生成一段老人打太极的视频吧", "files": []},
    {"text": "根据我的病历，给我一个合理的治疗方案", "files": []},
    {"text": "根据知识库介绍一下常见疾病", "files": []},
    {"text": "根据知识图谱告诉我糖尿病人适合吃的食物有哪些？", "files": []},
]


# 构建 Gradio 界面
with gr.Blocks(css=MEDICAL_THEME_CSS) as demo:
    # 标题和描述
    gr.Markdown("# 「赛博华佗」", elem_id="app-title")

    with gr.Row(elem_id="theme-toggle-row"):
        with gr.Column(scale=4, min_width=280):
            theme_mode = gr.Radio(
                choices=list(THEME_MODE_OPTIONS),
                value=THEME_MODE_SYSTEM,
                label="主题模式",
                elem_id="theme-mode",
                interactive=True,
            )

    # 创建聊天布局
    with gr.Row():
        with gr.Column(scale=10):
            chatbot = gr.Chatbot(
                height=600,
                avatar_images=AVATAR,
                show_copy_button=True,
                bubble_full_width=False,
                sanitize_html=False,
                elem_id="cyber-chatbot",
                latex_delimiters=[
                    {"left": "\\(", "right": "\\)", "display": True},
                    {"left": "\\[", "right": "\\]", "display": True},
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": True},
                ],
                placeholder="\n## 欢迎与我对话\n————赛博华佗将基于你提供的信息给出结构化健康建议",
            )

    with gr.Row(elem_id="guide-row"):
        guide_buttons = []
        for label, _ in QUICK_GUIDE_PROMPTS:
            guide_buttons.append(gr.Button(label, elem_classes=["guide-pill"], size="sm"))

    with gr.Row():
        with gr.Column(scale=9):
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="输入消息或上传文件...",
                show_label=False,
                elem_id="cyber-input",
            )
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                label="录音输入",
                visible=False,
                type="filepath",
            )
        with gr.Column(scale=1):
            clear = gr.ClearButton([chatbot, chat_input, audio_input], value="清除记录")
            toggle_voice_button = gr.Button("语音对话模式", visible=True)
            toggle_text_button = gr.Button("文本交流模式", visible=False)
            submit_audio_button = gr.Button("发送", visible=False)

    with gr.Row() as example_row:
        example_component = gr.Examples(
            examples=examples, inputs=chat_input, visible=True, examples_per_page=15
        )

    gr.HTML(MEDICAL_DISCLAIMER_HTML)

    demo.load(
        fn=normalize_theme_mode,
        inputs=[theme_mode],
        outputs=[theme_mode],
        js=THEME_BOOTSTRAP_JS,
        queue=False,
        show_progress="hidden",
    )
    theme_mode.change(
        fn=normalize_theme_mode,
        inputs=[theme_mode],
        outputs=[theme_mode],
        js=THEME_CHANGE_JS,
        queue=False,
        show_progress="hidden",
    )

    chat_input.submit(fn=grodio_view, inputs=[chatbot, chat_input], outputs=[chatbot])
    for guide_button, (_, prompt) in zip(guide_buttons, QUICK_GUIDE_PROMPTS):
        guide_button.click(
            fn=build_quick_prompt_handler(prompt),
            inputs=[chatbot],
            outputs=[chatbot],
        )

    # 切换按钮点击事件
    toggle_voice_button.click(
        fn=toggle_voice_mode,
        inputs=None,
        outputs=[
            chat_input,
            audio_input,
            toggle_voice_button,
            toggle_text_button,
            submit_audio_button,
        ],
    )

    toggle_text_button.click(
        fn=toggle_text_mode,
        inputs=None,
        outputs=[
            chat_input,
            audio_input,
            toggle_voice_button,
            toggle_text_button,
            submit_audio_button,
        ],
    )

    submit_audio_button.click(
        fn=gradio_audio_view, inputs=[chatbot, audio_input], outputs=[chatbot]
    )


# 启动应用
def start_gradio():
    demo.launch(server_port=10032, share=False)


if __name__ == "__main__":
    start_gradio()
