# app.py — Brand Fit Auditor (v3.2, sample image selectable)
# - Macro-first brand research + Refine pass
# - Executive Summary 표시
# - [Notes] (회색·소형 주석)로 표기
# - 점수 정합성 보정, 초기 CSS 오버레이 핫스팟 + 중복 제거
# - ✅ 샘플 이미지(sample_kimchitoktok.png) 기본 제공 & 선택 포함
# 실행: streamlit run app.py
# 필요: pip install -U google-genai streamlit beautifulsoup4 requests

import os, re, json, base64, math
from pathlib import Path
from typing import Optional, List, Tuple

import requests
import streamlit as st
from bs4 import BeautifulSoup
from requests.exceptions import SSLError

# Gemini SDK
from google import genai
from google.genai import types

# ===============================
# 0) API KEY (.env → ENV → secrets)
# ===============================
def _parse_env_file(path: str) -> dict:
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return out

def load_api_key() -> Optional[str]:
    if hasattr(st, "secrets"):
        v = st.secrets.get("GEMINI_API_KEY", None)
        if v: return v
    v = os.environ.get("GEMINI_API_KEY")
    if v: return v
    envmap = _parse_env_file(".env")
    v = envmap.get("GEMINI_API_KEY")
    if v:
        os.environ["GEMINI_API_KEY"] = v
        return v
    return None

API_KEY = load_api_key()
if not API_KEY:
    st.error("❌ GEMINI_API_KEY가 없습니다. .env 또는 환경변수/Streamlit secrets에 설정하세요.")
    st.stop()

# ===============================
# 1) Gemini client + helpers
# ===============================
@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key)

client = get_client(API_KEY)

def _gen_config():
    return types.GenerateContentConfig(
        response_modalities=["TEXT"],
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

def call_gemini_text(prompt: str, model: str) -> str:
    try:
        cfg = _gen_config()
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def call_gemini_mm(prompt: str, image_parts: List[types.Part], model: str) -> str:
    try:
        cfg = _gen_config()
        parts = [types.Part.from_text(text=prompt)] + (image_parts or [])
        resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def parse_json_or_fail(raw: str, fail_title: str) -> dict:
    try:
        s = raw.find("{"); e = raw.rfind("}")
        data = json.loads(raw[s:e+1]) if s != -1 and e != -1 and e > s else None
    except Exception:
        data = None
    if not data:
        st.error(f"{fail_title} — LLM JSON 파싱 실패")
        with st.expander("LLM 원문 보기"):
            st.code(raw)
        st.stop()
    return data

# ===============================
# 2) Crawl + pack builders
# ===============================
def fetch_html(url: str) -> Tuple[Optional[str], Optional[str]]:
    if not url: return None, "URL 비어있음"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return r.text, None
    except SSLError:
        try:
            r = requests.get(url, headers=headers, timeout=20, verify=False)
            r.raise_for_status()
            return r.text, "⚠️ SSL 인증서 검증 실패 → 보안 검증 생략"
        except Exception as e2:
            return None, f"크롤링 오류(SSL): {e2}"
    except Exception as e:
        return None, f"크롤링 오류: {e}"

def build_read_pack(html: str, max_body=14000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script","style","noscript","meta","iframe","svg"]): t.decompose()
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    heads = [h.get_text(" ", strip=True) for h in soup.find_all(["h1","h2","h3","h4"]) if h.get_text(strip=True)]
    emph  = [e.get_text(" ", strip=True) for e in soup.find_all(["strong","b","em","mark"]) if e.get_text(strip=True)]
    lis   = [li.get_text(" ", strip=True) for li in soup.find_all("li") if li.get_text(strip=True)]
    body  = soup.get_text(" ", strip=True)[:max_body]
    blocks = []
    if title: blocks.append(f"[TITLE]\n{title}")
    if heads: blocks.append("[HEADLINES]\n- " + "\n- ".join(dict.fromkeys(heads)))
    if emph:  blocks.append("[EMPHASIS]\n- " + "\n- ".join(dict.fromkeys(emph)))
    if lis:   blocks.append("[LIST]\n- " + "\n- ".join(lis[:300]))
    blocks.append("[BODY]\n" + body)
    return "\n\n".join(blocks)

@st.cache_data(show_spinner=False)
def wiki_summary(brand: str) -> str:
    def _get(lang: str) -> Optional[str]:
        try:
            q = requests.utils.quote(brand)
            s = requests.get(f"https://{lang}.wikipedia.org/w/rest.php/v1/search/title?q={q}&limit=1", timeout=10).json()
            if not s.get("pages"): return None
            title = s["pages"][0]["title"]
            j = requests.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}", timeout=10).json()
            return f"[WIKIPEDIA:{lang}/{title}]\n{(j.get('extract') or '').strip()}"
        except Exception:
            return None
    ko = _get("ko"); en = _get("en")
    return "\n\n".join([b for b in [ko, en] if b]) or ""

@st.cache_data(show_spinner=False)
def guess_brand_sources(brand: str, already: List[str]) -> List[str]:
    slug = re.sub(r"[^a-z0-9]", "", brand.lower())
    cands = []
    for base in [f"https://{slug}.com", f"https://www.{slug}.com", f"https://{slug}.co.kr", f"https://www.{slug}.co.kr"]:
        cands += [base, base+"/about", base+"/company", base+"/kr"]
    cands.append(f"https://www.instagram.com/{slug}")
    picked, seen = [], set(u.strip().lower() for u in already)
    for u in cands:
        if len(picked) >= 3: break
        if u.lower() in seen: continue
        html, err = fetch_html(u)
        if html: picked.append(u)
    return picked

# ===============================
# 3) Prompts (Macro-first + Refine)
# ===============================
BRAND_RESEARCH_PROMPT = """
당신은 시니어 브랜드 스트래티지스트다.

목표: 입력된 여러 출처(공식 사이트/회사 소개/보도자료/위키/공식 소셜 등)를 근거 삼아
브랜드를 **지나치게 미시적(특정 메뉴/프로모션/캠페인)** 으로 정의하지 말고
**상위 레벨(기업/마스터브랜드) 관점에서** 요약하라.

매크로 우선 원칙 (반드시 준수):
- 우선순위: 업(Industry) → 카테고리/핵심 제공가치 → 포지셔닝/차별점 → 주요 고객군/지역 → 시각/톤 특성.
- 개별 SKU·메뉴·한시적 캠페인은 ‘예시’로만 언급하고, notable_programs_or_subbrands에만 나열한다.
- “브랜드를 한 줄로 정의”할 때 특정 메뉴명이 주어가 되지 않도록 한다.
- 하나의 소스에 과적합되지 말고, 여러 출처의 공통분모를 우선 채택한다.

아래 **JSON만** 반환한다(필드 유지, 필요시 일부는 빈 문자열/배열 허용):

{
  "brand": "<브랜드명>",
  "category": "<상위 업/카테고리 예: 글로벌 패스트푸드 프랜차이즈, 스포츠웨어, 소비자가전 등>",
  "brand_scope": "corporate|masterbrand|product_line",
  "granularity": "macro|meso|micro",
  "executive_summary": "상위 관점 3~5문장 요약(업/규모/핵심가치/차별점/대표 제공물)",
  "primary_offerings": ["제품/서비스 대분류(예: '패스트푸드', '스마트폰', '스포츠웨어' 등)", ""],
  "brand_identity": {
    "positioning": "",
    "values": ["", ""],
    "tone_voice": ["", ""],
    "visual_cues": ["colors / imagery style / logo rules 등 상위 표현"]
  },
  "target_audience": ["", ""],
  "market_perception": {
    "top_keywords": ["", ""],
    "explanation": "소비자/미디어 관점의 상위 인식(지엽적 메뉴명 중심 금지)",
    "notes": ""
  },
  "notable_programs_or_subbrands": ["(있다면) 하위 프로그램/서브브랜드 3개 이내"],
  "evidence_notes": "가장 신뢰도 높은 출처에 기반한 근거 요약 2~4문장",
  "confidence": 0.0
}

출력 규칙:
- granularity는 원칙적으로 "macro"여야 한다(기업/마스터브랜드 관점).
- primary_offerings/keywords에는 특정 SKU/메뉴명을 쓰지 말고 ‘범주’로 작성.
- notable_programs_or_subbrands에만 개별 프로그램/메뉴/캠페인을 넣는다.
"""

REFINE_BRAND_RESEARCH_PROMPT = """
아래 초기 결과가 지나치게 미시적이므로, 같은 증거를 사용하되
**기업/마스터브랜드 관점의 'macro' 수준**으로 다시 요약해라.
JSON 스키마와 규칙은 기존 BRAND_RESEARCH_PROMPT와 동일하며,
반드시 granularity="macro"로 설정한다. SKU/단일 메뉴명 중심 서술 금지.

[초기 응답 JSON]을 참고하되, notable_programs_or_subbrands 필드로만
개별 프로그램/메뉴를 분리해 명시하고 본문 요약과 category/primary_offerings에는
상위 범주만 사용하라.

반환은 JSON만.
"""

FIT_EVAL_PROMPT = """
당신은 Brand Guardianship 심사위원이다.
중요 규칙:
- dim.score는 0~100 정수.
- overall_score = round(mean([세 dim score]))
- verdict:
  80~100: "Strong fit"
  60~79 : "Good fit"
  40~59 : "Borderline"
  0~39  : "Misaligned"
JSON ONLY:
{
  "overall_score": 0, "verdict": "",
  "dimensions": [
    {"name":"Tone & Voice","score":0,"rationale":""},
    {"name":"Visual Identity","score":0,"rationale":""},
    {"name":"Brand-Product Relevance","score":0,"rationale":""}
  ],
  "copy_suggestions":[{"before":"","after":"","reason":""}],
  "cta_proposals":[{"cta":"","expected_effect":""}],
  "image_feedback":[
    {"index":1,"notes":"","risks":[""],"suggested_edits":[""],
     "hotspots":[
       {"shape":"circle","cx":0.72,"cy":0.40,"r":0.08,"label":"","risks":[""],"suggested_edits":[""]},
       {"shape":"rect","x":0.10,"y":0.25,"w":0.18,"h":0.10,"label":"","risks":[""],"suggested_edits":[""]}
     ]}
  ],
  "reasoning_notes":""
}
좌표: 0~1 정규화. label/risks/edits에는 번호 문자 넣지 마라(숫자 표시는 UI가 처리).
"""

# ===============================
# 4) Styles (카드/배지/차트/핫스팟)
# ===============================
CARD_CSS = """
<style>
:root{--card-bg:#f8fafc;--subcard-bg:#f3f4f6;--bar-bg:#e2e8f0;--bar-fill:#2563eb;--danger:#dc2626;}
.section-sep{border:0;border-top:1px solid #e5e7eb;margin:18px 0}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:10px;background:var(--card-bg);margin:10px 0;overflow-wrap:anywhere}
.subcard{border:1px solid #e5e7eb;border-radius:12px;padding:10px;background:var(--subcard-bg);margin:10px 0}
.card h4{margin:0 0 6px 0}
.meta{color:#6b7280;font-size:12px}
.note-muted{font-size:12px;color:#6b7280;margin:6px 0 10px 0}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:13px;color:#fff}
.badge.big{padding:6px 14px;font-size:15px;font-weight:800;}
.badge.gray{background:#9ca3af;color:#fff}
.meta-badges{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px 0}
.tag{display:inline-block;background:#e5e7eb;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:700;color:#374151}

.score-text{font-weight:800;font-size:22px}

.dimrow{display:flex;align-items:center;gap:14px;margin:8px 0}
.dimname{width:220px;font-weight:700}
.dimwrap{width:45%}
.dimbar{position:relative;height:18px;background:var(--bar-bg);border-radius:10px;overflow:hidden}
.dimbar>span{display:block;height:100%;background:var(--bar-fill)}
.dimbar::after{content:"";position:absolute;inset:0;background-image:repeating-linear-gradient(to right,rgba(100,116,139,.4) 0,rgba(100,116,139,.4) .5px,transparent .5px,transparent 20%)}
.dimscore{width:84px;text-align:right;font-weight:800;font-size:16px}

.rationale{color:#111827;font-size:14px;margin-top:8px}
.reasoning-hero{margin-top:6px;font-size:15px;color:#111827;font-weight:600}

.caps{display:grid;grid-template-columns:1fr;gap:10px}
.chipline{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px}
.chiplabel{display:inline-block;background:#e2e8f0;border-radius:999px;padding:2px 8px;font-size:12px;font-weight:700;margin-right:8px}

.copy-grid{display:grid;grid-template-columns:1fr 40px 1fr;gap:10px;align-items:center}
.copy-box{border:1px solid #e5e7eb;border-radius:10px;padding:10px;background:#fff}
.copy-arrow{text-align:center;font-weight:700}
.reason-title{font-weight:700;font-size:14px;color:#111827;margin-top:8px}
.reason-block{margin-bottom:18px}

.preview-wrap{position:relative;width:100%}
.preview-img{width:100%;max-width:100%;height:auto;display:block;border-radius:8px;border:1px solid #e5e7eb}

/* Hotspots — 초기 버전 스타일 */
.hotspot{position:absolute;border:4px solid var(--danger);box-shadow:0 0 0 4px rgba(220,38,38,0.15) inset;}
.hotspot.circle{border-radius:9999px;}
.hotspot .hs-num{
  position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
  background:var(--danger);color:#fff;font-weight:800;font-size:14px;border-radius:9999px;padding:2px 6px;line-height:1;
}
.hotspot:hover::after{
  content: attr(data-tip); position:absolute; left:50%; top:100%; transform: translate(-50%, 8px);
  background:#111827; color:#fff; font-size:12px; padding:6px 8px; border-radius:6px; white-space:normal; max-width:260px; z-index:3;
}
.hotspot:hover::before{
  content:""; position:absolute; left:50%; top:100%; transform: translate(-50%, 2px);
  border:6px solid transparent; border-bottom:0; border-top-color:#111827; z-index:3;
}

.anno{color:#111827;font-size:14px}
.anno li{margin-bottom:4px}
</style>
"""

# ===============================
# 5) Utils
# ===============================
def esc(s: str) -> str:
    s = str(s or ""); return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def attr_esc(s: str) -> str:
    return esc(s).replace('"', "&quot;").replace("'", "&#39;")

CIRCLED_RANGE = r"[\u2460-\u2473\u24F5-\u24FE\u2776-\u277F]"
def strip_circled(text: str) -> str:
    if not text: return ""
    t = re.sub(CIRCLED_RANGE, "", str(text))
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def to_image_part(up) -> Optional[types.Part]:
    if not up: return None
    try:
        data = up.read(); up.seek(0)
        mime = up.type or "application/octet-stream"
        return types.Part.from_bytes(data=data, mime_type=mime)
    except Exception:
        return None

def uploaded_to_data_uri(up) -> Optional[str]:
    if not up: return None
    try:
        data = up.read(); up.seek(0)
        mime = up.type or "image/png"
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def valid_dims(items: list) -> list:
    allowed = {"Tone & Voice", "Visual Identity", "Brand-Product Relevance"}
    out = []
    for d in items or []:
        if isinstance(d, dict) and isinstance(d.get("name"), str) and d["name"] in allowed:
            try:
                score = float(d.get("score", 0))
                out.append({"name": d["name"], "score": max(0, min(100, int(round(score)))), "rationale": d.get("rationale","")})
            except Exception:
                continue
    return out

def score_to_hsl(score: int) -> str:
    hue = max(0, min(120, int((score/100)*120)))  # 0 red ~ 120 green
    return f"hsl({hue}, 70%, 40%)"

def circled(n: int) -> str:
    circ = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩","⑪","⑫","⑬","⑭","⑮","⑯","⑰","⑱","⑲","⑳"]
    return circ[n-1] if 1 <= n <= len(circ) else f"({n})"

def compute_verdict(score: int) -> str:
    if score >= 80: return "Strong fit"
    if score >= 60: return "Good fit"
    if score >= 40: return "Borderline"
    return "Misaligned"

def reconcile_scores(fit: dict) -> dict:
    dims = valid_dims(fit.get("dimensions"))
    if dims:
        avg = int(round(sum(d["score"] for d in dims)/len(dims)))
        fit["dimensions"] = dims
        fit["overall_score"] = avg
        fit["verdict"] = compute_verdict(avg)
    else:
        fit["overall_score"] = max(0, min(100, int(fit.get("overall_score", 0) or 0)))
        fit["verdict"] = compute_verdict(fit["overall_score"])
    return fit

# --- Hotspot dedupe/merge (겹침 제거) ---
def _bbox(h: dict) -> Tuple[float,float,float,float]:
    if (h.get("shape") or "circle").lower() == "rect":
        x = float(h.get("x",0)); y=float(h.get("y",0)); w=float(h.get("w",0)); hgt=float(h.get("h",0))
        return (x, y, x+w, y+hgt)
    cx=float(h.get("cx",0.5)); cy=float(h.get("cy",0.5)); r=float(h.get("r",0.1))
    return (cx-r, cy-r, cx+r, cy+r)

def _area(b): 
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def _iou(b1,b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1]); ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    iw=max(0.0, ix2-ix1); ih=max(0.0, iy2-iy1)
    inter=iw*ih; union=_area(b1)+_area(b2)-inter
    return inter/union if union>0 else 0.0

def _centerdist(b1,b2):
    c1=((b1[0]+b1[2])/2, (b1[1]+b1[3])/2); c2=((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
    return math.hypot(c1[0]-c2[0], c1[1]-c2[1])

def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    if not out.get("label") and b.get("label"): out["label"] = b["label"]
    out["risks"] = [*{*(out.get("risks") or []), *(b.get("risks") or [])}]
    out["suggested_edits"] = [*{*(out.get("suggested_edits") or [])}, *(b.get("suggested_edits") or [])]
    return out

def dedupe_hotspots(hotspots: list) -> list:
    hs = [h for h in hotspots or [] if isinstance(h, dict)]
    hs_sorted = sorted(hs, key=lambda h: _area(_bbox(h)), reverse=True)
    kept = []
    for h in hs_sorted:
        b = _bbox(h); merged=False
        for i, k in enumerate(kept):
            bk = _bbox(k)
            if _iou(b, bk) > 0.55 or _centerdist(b, bk) < 0.12:
                kept[i] = _merge(k, h); merged=True; break
        if not merged:
            hh = dict(h)
            for key in ["x","y","w","h","cx","cy","r"]:
                if key in hh:
                    try:
                        v = float(hh[key]); hh[key] = max(0.0, min(1.0, v))
                    except Exception: pass
            kept.append(hh)
    return kept[:12]

# ===============================
# 6) UI
# ===============================
st.set_page_config(page_title="Brand Fit Auditor", page_icon="🧭", layout="centered")
st.title("🧭 Brand Fit Auditor")
st.markdown(CARD_CSS, unsafe_allow_html=True)

with st.expander("도움말", expanded=False):
    st.markdown(
        "Brand Fit Auditor는 광고/마케팅에 활용되는 소재(이미지/텍스트 등)가 브랜드의 전체적인 정체성 및 이미지와 적합한지를 다각도로 검증해주는 AI Agent입니다.\n\n"
        "**이런 분들에게 추천드립니다.**\n\n"
        "‍🤦‍♂ 짧은 제작 리드타임 안에 배너·영상 썸네일을 대량 제작 및 검수하는 데 어려움을 겪는 퍼포먼스 마케터/그로스/소셜 마케터!\n\n"
        "🤦️ 팀원 또는 파트너/리셀러가 만든 공동 마케팅 소재의 브랜딩 일탈을 모니터링하는 데 어려움을 겪는 마케팅 매니저/채널 세일즈 매니저!\n\n"
        "🤦‍♀️ 여러 마케팅 산출물을 일관성 있게 품질관리하는 데 어려움을 겪는 브랜드 매니저/거버넌스 담당자!\n"
    )

# (요청) 모델 선택 UI 삭제 → 내부 고정값 사용
model = "gemini-2.5-flash"

# (요청) 기본값 설정
brand = st.text_input("1) 내 브랜드명", value="LG", placeholder="예: LG, Samsung, Nike ...")
urls  = st.text_input("브랜드 참고 URL (최대 3개, 쉼표로 구분)", value="https://www.lge.co.kr/home", placeholder="예: https://www.lge.co.kr, https://www.instagram.com/lg ...")
st.caption("브랜드 공식 홈페이지 또는 브랜드의 Identity를 잘 보여줄 수 있는 웹페이지의 URL을 입력해주세요.")

copy_txt = st.text_area(
    "마케팅/광고에 사용할 카피라이팅 및 캡션을 입력해주세요",
    value="김치톡톡 지금 사야 제맛. 김치톡톡 런칭 혜택전. 미색 생활을 완성하는 남다른 보관 방법!",
    placeholder="카피/캡션/해시태그",
    height=120
)

# ========= ✅ 샘플 이미지 기본 제공(경로 탐색 강화) =========
def find_sample_file() -> Optional[Path]:
    """
    sample_kimchitoktok.png / .PNG 를 아래 경로에서 순서대로 탐색:
    1) app.py 폴더, 2) 현재 작업 디렉토리, 3) ./image 폴더
    """
    names = ["sample_kimchitoktok.png", "sample_kimchitoktok.PNG"]
    candidates: List[Path] = []

    try:
        here = Path(__file__).resolve().parent
        candidates += [here / n for n in names]
        candidates += [here / "image" / n for n in names]
    except Exception:
        pass

    cwd = Path(os.getcwd())
    candidates += [cwd / n for n in names]
    candidates += [cwd / "image" / n for n in names]

    for p in candidates:
        if p.is_file():
            return p
    return None

sample_file = find_sample_file()
use_sample = False

if sample_file:
    with st.container():
        st.markdown("**예시 이미지 사용(선택 사항)**")
        cols_s = st.columns([1, 2])
        with cols_s[0]:
            try:
                b = sample_file.read_bytes()
                b64 = base64.b64encode(b).decode("utf-8")
                st.image(f"data:image/png;base64,{b64}",
                         caption=str(sample_file.name),
                         use_container_width=True)
            except Exception:
                st.info("샘플 이미지 미리보기를 로드하지 못했습니다.")
        with cols_s[1]:
            use_sample = st.checkbox(
                "샘플 이미지를 분석에 포함하기",
                value=True,
                help=f"경로: {sample_file}"
            )
else:
    st.caption(
        "※ 샘플 이미지가 보이지 않나요? 아래 경로 중 하나에 "
        "`sample_kimchitoktok.png` 파일을 두세요.\n"
        f"- app.py와 같은 폴더\n"
        f"- 현재 작업 디렉토리: {os.getcwd()}\n"
        f"- 위 경로의 `image/` 폴더 내부"
    )
# =========================================================

imgs = st.file_uploader(
    "마케팅/광고에 사용할 소재 이미지를 최대 3장까지 업로드 해주세요.",
    type=["png","jpg","jpeg","webp"],
    accept_multiple_files=True
)

# (요청) 버튼 문구 변경
go = st.button("분석 시작", type="primary")

# ===============================
# 7) Run
# ===============================
if go:
    if not brand:
        st.warning("브랜드명을 입력하세요."); st.stop()
    if not copy_txt and not imgs and not (use_sample and sample_file and sample_file.is_file()):
        st.warning("텍스트 또는 이미지를 최소 1개 이상 제공하세요."); st.stop()

    # Evidence 수집
    with st.spinner("AI가 브랜드에 대해 리서치 하는 중"):
        packs, warnings = [], []
        url_list = [u.strip() for u in urls.split(",") if u.strip()] if urls.strip() else []
        url_list = url_list[:3]
        extra_sources = guess_brand_sources(brand, url_list)
        for u in url_list + extra_sources:
            html, warn = fetch_html(u)
            if html: packs.append(f"[SOURCE]\n{u}\n\n{build_read_pack(html)}")
            if warn: warnings.append(f"{u} → {warn}")
        wiki = wiki_summary(brand)
        if wiki: packs.append(wiki)
        evidence_text = ("\n\n---\n\n").join(packs) if packs else "(증거 부족)"
    for w in warnings: st.warning(w)

    # ① 브랜드 리서치 (Macro-first + Refine)
    with st.spinner("AI가 브랜드에 대해 리서치 하는 중"):
        br_prompt = f"{BRAND_RESEARCH_PROMPT}\n\n[브랜드명]\n{brand}\n\n[증거 텍스트]\n{evidence_text}"
        br_raw = call_gemini_text(br_prompt, model=model)
        br_json = parse_json_or_fail(br_raw, "브랜드 리서치")

    need_refine = (br_json.get("granularity","").lower() != "macro") or not (br_json.get("category") or "").strip()
    if need_refine:
        refine_prompt = (
            f"{REFINE_BRAND_RESEARCH_PROMPT}\n\n"
            f"[브랜드명]\n{brand}\n\n"
            f"[증거 텍스트]\n{evidence_text}\n\n"
            f"[초기 응답 JSON]\n{json.dumps(br_json, ensure_ascii=False)}"
        )
        br_raw2 = call_gemini_text(refine_prompt, model=model)
        br_json2 = parse_json_or_fail(br_raw2, "브랜드 리서치(재정렬)")
        br_json = br_json2

    # --- ① 브랜드 리서치 요약 UI ---
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>① 브랜드 리서치 요약</h4>", unsafe_allow_html=True)

    st.write(f"**브랜드:** {br_json.get('brand') or brand}")

    # 메타 배지
    badges = []
    if br_json.get("category"):
        badges.append(f"<span class='tag'>Category: {esc(br_json['category'])}</span>")
    if br_json.get("brand_scope"):
        badges.append(f"<span class='tag'>Scope: {esc(br_json['brand_scope'])}</span>")
    if br_json.get("granularity"):
        badges.append(f"<span class='tag'>Granularity: {esc(br_json['granularity'])}</span>")
    if badges:
        st.markdown("<div class='meta-badges'>" + " ".join(badges) + "</div>", unsafe_allow_html=True)

    # [Notes] (회색·소형) + Executive Summary
    if br_json.get("evidence_notes"):
        st.write(f"<div class='note-muted'>[Notes] {esc(br_json['evidence_notes'])}</div>", unsafe_allow_html=True)
    if br_json.get("executive_summary"):
        st.write(f"<div class='rationale'><b>Executive Summary</b><br>{esc(br_json['executive_summary'])}</div>", unsafe_allow_html=True)

    bi = br_json.get("brand_identity", {}) or {}
    mp = br_json.get("market_perception", {}) or {}
    prim = br_json.get("primary_offerings") or []
    subs = br_json.get("notable_programs_or_subbrands") or []
    mp_keywords = ", ".join(mp.get("top_keywords") or []) or "—"
    mp_expl = mp.get("explanation") or mp.get("notes") or "—"

    chips = []
    if prim:
        chips.append(f"<div class='chipline'><span class='chiplabel'>Primary Offerings</span>{esc(', '.join(prim))}</div>")
    chips.append(f"<div class='chipline'><span class='chiplabel'>Positioning</span>{esc(bi.get('positioning') or '정보 부족')}</div>")
    chips.append(f"<div class='chipline'><span class='chiplabel'>Values</span>{esc(', '.join(bi.get('values') or []) or '정보 부족')}</div>")
    chips.append(f"<div class='chipline'><span class='chiplabel'>Tone &amp; Voice</span>{esc(', '.join(bi.get('tone_voice') or []) or '정보 부족')}</div>")
    chips.append(f"<div class='chipline'><span class='chiplabel'>Visual Cues</span>{esc(', '.join(bi.get('visual_cues') or []) or '정보 부족')}</div>")
    chips.append(f"<div class='chipline'><span class='chiplabel'>Target Audience</span>{esc(', '.join(br_json.get('target_audience') or []) or '정보 부족')}</div>")
    chips.append(f"<div class='chipline'><span class='chiplabel'>Market Perception</span>{esc(mp_expl)}<br>· 주요 키워드: {esc(mp_keywords)}</div>")
    if subs:
        chips.append(f"<div class='chipline'><span class='chiplabel'>Notables</span>{esc(', '.join(subs))}</div>")

    st.markdown("<div class='caps'>" + "".join(chips) + "</div></div>", unsafe_allow_html=True)

    # 이미지 준비
    image_parts, data_uris = [], []

    # ✅ 샘플 이미지 우선 포함 (최대 3장 제한 안에서)
    if use_sample and sample_file and sample_file.is_file():
        try:
            sb = sample_file.read_bytes()
            image_parts.append(types.Part.from_bytes(data=sb, mime_type="image/png"))
            data_uris.append("data:image/png;base64," + base64.b64encode(sb).decode("utf-8"))
        except Exception:
            st.info("샘플 이미지를 불러오지 못했습니다.")

    # 업로드 이미지 포함 (총 3장 제한)
    if imgs:
        for up in imgs:
            if len(image_parts) >= 3:
                break
            p = to_image_part(up)
            if p:
                image_parts.append(p)
                data_uris.append(uploaded_to_data_uri(up))

    # ② 적합성 평가
    with st.spinner("AI가 브랜드 적합성을 평가 중..."):
        fit_ctx = json.dumps(br_json, ensure_ascii=False)
        mm_prompt = f"{FIT_EVAL_PROMPT}\n\n[브랜드 리서치 JSON]\n{fit_ctx}\n\n[광고 텍스트]\n{copy_txt.strip() or '(제공 없음)'}\n\n[이미지] 업로드 순서 기준 1부터."
        fit_raw = call_gemini_mm(mm_prompt, image_parts, model=model) if image_parts else call_gemini_text(mm_prompt, model=model)
        fit_json = parse_json_or_fail(fit_raw, "적합성 평가")

    fit_json = reconcile_scores(fit_json)

    # --- ② 결과 UI ---
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>② 브랜드 적합성 평가 결과</h4>", unsafe_allow_html=True)
    overall = int(fit_json.get("overall_score", 0))
    verdict = fit_json.get("verdict") or "—"
    st.write(
        f"<span class='score-text'>**Overall Score: {overall}/100**</span> "
        f"<span class='badge big' style='background:{score_to_hsl(overall)}'>{esc(verdict)}</span>",
        unsafe_allow_html=True
    )
    if fit_json.get("reasoning_notes"):
        st.markdown(f"<div class='reasoning-hero'>[평가 근거] {esc(strip_circled(fit_json['reasoning_notes']))}</div>", unsafe_allow_html=True)

    for d in fit_json.get("dimensions", []):
        rationale = strip_circled(d.get("rationale"))
        st.markdown(
            "<div class='subcard'><div class='dimrow'>"
            f"<div class='dimname'>{esc(d['name'])}</div>"
            "<div class='dimwrap'><div class='dimbar'>"
            f"<span style='width:{d['score']}%'></span></div></div>"
            f"<div class='dimscore'>{d['score']}/100</div></div>"
            f"<div class='rationale'>{esc(rationale)}</div></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- ③ 수정 제안 UI ---
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>③ 마케팅 소재 수정 제안</h4>", unsafe_allow_html=True)
    cs = fit_json.get("copy_suggestions") or []
    if cs:
        st.write("**카피라이팅 수정 제안**")
        for c in cs[:5]:
            before = strip_circled((c.get("before","") or "").strip())
            after  = strip_circled((c.get("after","") or "").strip())
            reason = strip_circled((c.get("reason","") or "").strip())
            inner = (
                "<div class='copy-grid'>"
                f"<div class='copy-box'><b>Before</b><br>{esc(before)}</div>"
                "<div class='copy-arrow'>→</div>"
                f"<div class='copy-box'><b>After</b><br><b>{esc(after)}</b></div>"
                "</div>"
            )
            if reason:
                inner += "<div class='reason-title'>제안 이유</div>"
                inner += f"<div class='reason-block'>{esc(reason)}</div>"
            st.markdown(f"<div class='subcard'>{inner}</div>", unsafe_allow_html=True)

    ctas = fit_json.get("cta_proposals") or []
    if ctas:
        st.write("**CTA (Call To Action) 제안**")
        for item in ctas[:6]:
            cta = strip_circled((item.get("cta") or "").strip())
            why = strip_circled((item.get("expected_effect") or "").strip())
            st.markdown(f"- **{esc(cta)}** — <small>{esc(why)}</small>", unsafe_allow_html=True)

    # --- 이미지 피드백 (초기 CSS 오버레이 + 중복 제거) ---
    imgs_feedback = fit_json.get("image_feedback") or []
    if imgs_feedback:
        st.write("**이미지 피드백**")
        for it in imgs_feedback[:3]:
            idx = it.get("index", 1)
            notes = strip_circled(it.get("notes","").strip())
            img_risks = [strip_circled(r) for r in (it.get("risks") or []) if r]
            img_edits = [strip_circled(e) for e in (it.get("suggested_edits") or []) if e]
            hotspots = dedupe_hotspots(it.get("hotspots") or [])

            img_src = None
            if imgs and 1 <= idx <= len(imgs): img_src = uploaded_to_data_uri(imgs[idx-1])
            elif data_uris and 1 <= idx <= len(data_uris): img_src = data_uris[idx-1]

            st.markdown("<div class='subcard'>", unsafe_allow_html=True)

            overlay_html = "<div class='preview-wrap'>"
            if img_src:
                overlay_html += f"<img src='{img_src}' class='preview-img'/>"
                for j, hp in enumerate(hotspots[:20], start=1):
                    num = circled(j)
                    shape = (hp.get("shape") or "circle").lower()
                    label = strip_circled(hp.get("label") or "")
                    tip = attr_esc(f"{num} {label}")
                    if shape == "circle":
                        cx=float(hp.get("cx",0.5)); cy=float(hp.get("cy",0.5)); r=float(hp.get("r",0.08))
                        left=max(0.0, cx-r)*100; top=max(0.0, cy-r)*100; size=min(1.0, r*2)*100
                        overlay_html += (
                            f"<div class='hotspot circle' data-tip=\"{tip}\" "
                            f"style='left:{left:.2f}%;top:{top:.2f}%;width:{size:.2f}%;height:{size:.2f}%;'>"
                            f"<div class='hs-num'>{num}</div></div>"
                        )
                    else:
                        x=float(hp.get("x",0)); y=float(hp.get("y",0)); w=float(hp.get("w",0)); h=float(hp.get("h",0))
                        overlay_html += (
                            f"<div class='hotspot' data-tip=\"{tip}\" "
                            f"style='left:{x*100:.2f}%;top:{y*100:.2f}%;width:{w*100:.2f}%;height:{h*100:.2f}%;border-radius:10px;'>"
                            f"<div class='hs-num'>{num}</div></div>"
                        )
            overlay_html += "</div>"
            st.markdown(overlay_html, unsafe_allow_html=True)

            if notes:
                st.markdown(f"<div class='rationale'><b>요약:</b> {esc(notes)}</div>", unsafe_allow_html=True)

            lines=[]
            for j, hp in enumerate(hotspots[:20], start=1):
                label = esc(strip_circled(hp.get("label") or ""))
                h_risks = [esc(strip_circled(r)) for r in (hp.get("risks") or []) if r]
                h_edits = [esc(strip_circled(e)) for e in (hp.get("suggested_edits") or []) if e]
                line = f"{j}. <b>{label}</b>"
                if h_risks: line += " — 위험요소: " + "; ".join(h_risks)
                if h_edits: line += " — 수정 제안: " + "; ".join(h_edits)
                lines.append(line)
            if lines:
                st.markdown("<div class='anno'><ul>" + "".join([f"<li>{l}</li>" for l in lines]) + "</ul></div>", unsafe_allow_html=True)
            else:
                if img_risks:
                    st.markdown("<div class='anno'><b>위험요소</b></div>", unsafe_allow_html=True)
                    st.markdown("<div class='anno'><ul>" + "".join([f"<li>{esc(r)}</li>" for r in img_risks]) + "</ul></div>", unsafe_allow_html=True)
                if img_edits:
                    st.markdown("<div class='anno'><b>수정 제안</b></div>", unsafe_allow_html=True)
                    st.markdown("<div class='anno'><ul>" + "".join([f"<li>{esc(e)}</li>" for e in img_edits]) + "</ul></div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 결과 다운로드
    st.download_button(
        "JSON 결과 다운로드",
        data=json.dumps({"brand_research": br_json, "fit_evaluation": fit_json}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="brand_fit_result.json",
        mime="application/json"
    )

    st.success("✅ 분석 완료")
