from Internet.Internet_prompt import extract_question
from Internet.file_utils import safe_filename
from Internet.retrieve_Internet import retrieve_html
from Internet.search_utils import (
    build_snippet_context,
    choose_effective_search_question,
    extract_real_url,
    rank_hits_by_query,
)
from client.clientfactory import Clientfactory
from env import get_app_root
from utils.console import safe_print

import os
import re
import requests
import shutil
import threading
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

_SAVE_PATH = os.path.join(get_app_root(), "data/cache/internet")
_DEFAULT_SNIPPET_MAX_ITEMS = 6


def build_internet_prompt(question: str, html_context: str, snippet_context: str) -> str:
    if html_context:
        return (
            f"根据你现有的知识，辅助以搜索到的文件资料：\n{html_context}\n"
            f"回答问题：\n{question}\n尽可能多的覆盖到文件资料"
        )

    if snippet_context:
        return (
            f"根据你现有的知识，辅助以下联网搜索摘要资料：\n{snippet_context}\n"
            f"回答问题：\n{question}\n"
            "若摘要与常识冲突，请标注不确定。"
        )

    return question


def _record_search_hit(
    hits,
    hits_lock,
    links,
    links_lock,
    title: str,
    link: str,
    snippet: str,
):
    if not link:
        return

    hit = {
        "title": (title or "").strip(),
        "link": (link or "").strip(),
        "snippet": (snippet or "").strip(),
    }

    with hits_lock:
        hits.append(hit)

    with links_lock:
        if hit["link"] not in links:
            links[hit["link"]] = hit["title"] or hit["link"]


def InternetSearchChain(question, history):
    user_question = question

    if os.path.exists(_SAVE_PATH):
        shutil.rmtree(_SAVE_PATH)
    os.makedirs(_SAVE_PATH, exist_ok=True)

    extracted_question = extract_question(question, history)
    whole_question = choose_effective_search_question(
        original_question=user_question,
        extracted_question=extracted_question,
    )
    question_list = [q.strip() for q in re.split(r"[;；]", whole_question) if q.strip()]
    if not question_list:
        question_list = [user_question]

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    threads = []
    links = {}
    links_lock = threading.Lock()
    hits = []
    hits_lock = threading.Lock()

    for sub_question in question_list:
        thread = threading.Thread(
            target=search_bing,
            args=(sub_question, links, links_lock, hits, hits_lock, 3),
        )
        threads.append(thread)
        thread.start()

        thread = threading.Thread(
            target=search_baidu,
            args=(sub_question, links, links_lock, hits, hits_lock, 3),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    ranked_hits = rank_hits_by_query(
        hits,
        query=user_question,
        max_items=_DEFAULT_SNIPPET_MAX_ITEMS,
    )
    snippet_context = build_snippet_context(ranked_hits, max_items=_DEFAULT_SNIPPET_MAX_ITEMS)

    with links_lock:
        for hit in ranked_hits:
            link = hit.get("link")
            title = hit.get("title") or link
            if link and link not in links:
                links[link] = title

    retrieval_success = bool(snippet_context)
    html_context = ""

    if has_html_files(_SAVE_PATH):
        try:
            _docs, _context = retrieve_html(user_question)
            if _context.strip():
                html_context = _context
                retrieval_success = True
        except Exception as e:
            safe_print("Internet retrieval failed:", e)

    prompt = build_internet_prompt(
        question=user_question,
        html_context=html_context,
        snippet_context=snippet_context,
    )
    response = Clientfactory().get_client().chat_with_ai_stream(prompt)
    return response, links, retrieval_success


def search_bing(query, links, links_lock, hits, hits_lock, num_results=3):
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, compress",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0",
    }
    search_urls = [
        f"https://cn.bing.com/search?q={query}",
        f"https://www.bing.com/search?q={query}",
    ]

    for search_url in search_urls:
        saved_count = 0
        response = requests.get(search_url, headers=headers, verify=False, timeout=12)

        if response.status_code != 200:
            safe_print("Error:", response.status_code)
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for item in soup.find_all("li", class_="b_algo"):
            if saved_count >= num_results:
                break

            title_node = item.find("h2")
            if not title_node:
                continue
            link_node = title_node.find("a")
            if not link_node:
                continue

            title = title_node.get_text(" ", strip=True)
            raw_link = link_node["href"].split("#")[0]
            link = extract_real_url(raw_link)
            snippet_node = item.find("p")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""

            _record_search_hit(
                hits=hits,
                hits_lock=hits_lock,
                links=links,
                links_lock=links_lock,
                title=title,
                link=link,
                snippet=snippet,
            )

            try:
                page_response = requests.get(link, timeout=10, allow_redirects=True)
                if page_response.status_code != 200 or not page_response.text:
                    safe_print(
                        f"Failed to download {link}: Status code {page_response.status_code}"
                    )
                    continue

                filename = f"{_SAVE_PATH}/{safe_filename(title)}.html"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(page_response.text)
                saved_count += 1
                safe_print(f"Downloaded and saved: {link} as {filename}")
            except Exception as e:
                safe_print(f"Error downloading {link}: {e}")

        if saved_count < num_results:
            safe_print("访问bing失败，请检查网络代理")


def search_baidu(query, links, links_lock, hits, hits_lock, num_results=3):
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, compress",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0",
    }
    search_url = f"https://www.baidu.com/s?wd={query}"

    saved_count = 0
    response = requests.get(search_url, headers=headers, verify=False, timeout=12)

    if response.status_code != 200:
        safe_print("Error:", response.status_code)
        return

    soup = BeautifulSoup(response.text, "html.parser")
    for item in soup.find_all("div", class_="result"):
        if saved_count >= num_results:
            break
        try:
            title_node = item.find("h3")
            link_node = item.find("a")
            if not title_node or not link_node:
                continue

            title = title_node.get_text(" ", strip=True)
            link = link_node["href"].split("#")[0]
            snippet = item.get_text(" ", strip=True)

            _record_search_hit(
                hits=hits,
                hits_lock=hits_lock,
                links=links,
                links_lock=links_lock,
                title=title,
                link=link,
                snippet=snippet,
            )

            page_response = requests.get(link, timeout=10, allow_redirects=True)
            if page_response.status_code != 200 or not page_response.text:
                safe_print(
                    f"Failed to download {link}: Status code {page_response.status_code}"
                )
                continue

            filename = f"{_SAVE_PATH}/{safe_filename(title)}.html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(page_response.text)
            saved_count += 1
            safe_print(f"Downloaded and saved: {link} as {filename}")
        except Exception as e:
            safe_print(f"Error downloading {link}: {e}")

    if saved_count < num_results:
        safe_print("访问百度失败，请检查网络代理制")


def has_html_files(directory_path):
    if not os.path.exists(directory_path):
        return False

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".html"):
            return True
    return False
