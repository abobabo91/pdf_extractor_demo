#TODO


import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import os
import PyPDF2
from pdf2image import convert_from_bytes
import gc
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from io import BytesIO
import openai
from openai import OpenAI
import tiktoken
import re

import traceback, sys
import gc

def global_exception_handler(exc_type, exc_value, exc_traceback):
    st.error("Unhandled exception:")
    st.code("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = global_exception_handler


MODEL_PRICES = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 1.25, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1-nano": {"input": 0.05, "output": 0.40},
}

def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes DataFrame columns so they are Arrow-compatible for Streamlit."""
    if df is None or df.empty:
        return df
    df_fixed = df.copy()
    for col in df_fixed.columns:
        # minden object típusú oszlopot stringgé erőltetünk
        if df_fixed[col].dtype == "object":
            df_fixed[col] = df_fixed[col].astype(str)
    return df_fixed


def df_ready(df, required_cols=None):
    """Ellenőrzi, hogy a df nem üres és (opcionálisan) tartalmazza a kötelező oszlopokat."""
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return False
    if required_cols:
        return all(col in df.columns for col in required_cols)
    return True

def need_msg(missing_list):
    bullets = "\n".join([f"- {m}" for m in missing_list])
    st.warning(f"Az összefűzéshez a következők hiányoznak vagy üresek:\n{bullets}")

def count_tokens(text, model="gpt-4o"):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return len(tokens)

def replace_successive_duplicates(df, column_to_compare, columns_to_delete):
    result = df.copy()
    col = column_to_compare
    mask = result[col] == result[col].shift()
    for col in columns_to_delete:
        result.loc[mask, col] = np.nan
    return result

def extract_text_from_pdf(uploaded_file):
    import cv2
    from PIL import Image

    file_name = uploaded_file.name
    pdf_content = ""

    # 1) sima szövegkinyerés
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            pdf_content += page.extract_text() or ""
    except Exception as e:
        st.error(f"Hiba a(z) {file_name} fájl olvasásakor: {e}")
        return None

    # 2) OCR, ha túl kevés szöveg van
    if len(pdf_content.strip()) < 100:
        pdf_content = ""
        try:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()

            # determine number of pages
            num_pages = len(PyPDF2.PdfReader(BytesIO(file_bytes)).pages)

            progress = st.progress(0)
            for i in range(1, num_pages + 1):
                # higher DPI for sharper OCR
                images = convert_from_bytes(file_bytes, dpi=300, first_page=i, last_page=i)
#                images = convert_from_bytes(file_bytes, dpi=300, first_page=i, last_page=i, poppler_path = r"C:\poppler-24.08.0\Library\bin") #local)

                # --- OpenCV preprocessing with Otsu threshold ---
                img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # back to PIL for pytesseract
                img_pil = Image.fromarray(img)

                # OCR with Hungarian language, PSM 6 (single uniform block of text)
                custom_config = r'--psm 3'
                text = pytesseract.image_to_string(img_pil, lang="hun", config=custom_config)
                pdf_content += text + "\n"

                # memóriatisztítás
                del images, img, img_pil
                gc.collect()

                progress.progress(i / num_pages)

        except Exception as e:
            st.error(f"OCR hiba a(z) {file_name} fájlnál: {e}")
            return None

    # 3) hosszkorlátozás
    if len(pdf_content) > 300000:
        st.warning(file_name + " túl hosszú, csak az első 300000 karakter kerül feldolgozásra.")
        pdf_content = pdf_content[:300000]

    return pdf_content





def generate_gpt_prompt(text, file_name):
    """Generates a clear, structured GPT prompt for invoice data extraction."""
    return (
        f"You are given the extracted text of a Hungarian invoice PDF file named '{file_name}'. "
        "The PDF may contain multiple invoices merged together. "
        "Your task is to extract the following **9 data fields** for each invoice:\n\n"
        "1. Seller name (string)\n"
        "2. Buyer name (string)\n"
        "3. Invoice number (string)\n"
        "4. Invoice date (string, e.g. '2024.04.01')\n"
        "5. Total gross amount (integer)\n"
        "6. Total net amount (integer)\n"
        "7. VAT amount (integer)\n"
        "8. Currency (string: 'HUF' or 'EUR')\n"
        "9. Exchange rate (integer, use 1 if invoice is in HUF)\n\n"
        "10. Comments (string), any text that describes the business transaction the invoice contains (max 100 characters). If there is no comment, write an empty string.\n\n"
        "**Important formatting instructions:**\n"
        "- Use semicolon (`;`) to separate the 10 fields.\n"
        "- Use **one line per invoice**.\n"
        "- Do **not** include field numbers (e.g. '1)', '2)' etc.) in the output.\n"
        "- Write all numeric fields as plain integers (e.g. `1500000`).\n"
        "- **Do not use thousands separators** (e.g. `.`) or decimal commas (`,`) in the output and only output the integer part of the numbers.\n"
        "- Note: In Hungarian, decimal separators are commas (`,`) instead of dots (`.`) and thousand separators are dots (`.`)\n"
        "- The invoice number often appears in or matches the file name. Always first try to extract the invoice number from the text itself, but you can compare it to the file name too.\n"
        "- Do **not** include any explanation, headings, or extra text — just the data rows.\n\n"
        "Extracted text:\n"
        f"{text}"
    )


def extract_data_with_gpt(file_name, text, model_name):
    """A kiválasztott GPT modellel kinyeri a struktúrált adatokat a PDF szövegből."""
    gpt_prompt = generate_gpt_prompt(text, file_name)

    # --- Debug: show extracted text ---
#    with st.expander(f"📜 Kinyert szöveg – {file_name}"):
#        st.text_area("Extracted text", gpt_prompt[:10000], height=300)  # first 10k chars


    try:
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": gpt_prompt}],
            #max_completion_tokens=5000,
            timeout=30
        )

        raw_output = response.choices[0].message.content.strip()
        rows = raw_output.split("\n")

        parsed_rows = []
        for row in rows:
            parts = row.strip().split(";")
            if len(parts) != 10:
                st.warning(f"Hibás sor ({file_name}): {row}")
                continue

            cleaned_parts = [re.sub(r"^\s*\d+\)\s*", "", p.strip()) for p in parts]
            parsed_rows.append([file_name] + cleaned_parts)

        return parsed_rows, count_tokens(gpt_prompt)

    except Exception as e:
        st.error(f"A {model_name} feldolgozás sikertelen volt: {file_name} – {e}")
        return [], 0



def normalize_number(value):
    """Converts numeric-looking strings or floats to int. Removes all formatting."""
    try:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return int(round(value))
        # Remove thousands separators ('.' or ',' or space), allow decimals
        cleaned = str(value).replace(" ", "").replace(",", "").replace(".", "")
        return int(cleaned)
    except:
        return None


def compare_with_tolerance(val1, val2, tolerance=500):
    try:
        val1 = normalize_number(val1)
        val2 = normalize_number(val2)

        # Ha bármelyik hiányzik → "No Data"
        if val1 is None or val2 is None or pd.isna(val1) or pd.isna(val2):
            return "Nincs adat"

        return "Igen" if abs(val1 - val2) <= tolerance else "Nem"
    except Exception:
        return "Nincs adat"

    
    

def get_minta_amount(row, huf_col="Érték", eur_col="Érték deviza", currency_col="Devizanem"):
    """Returns the value in correct currency column based on Devizanem."""
    try:
        dev = str(row[currency_col]).strip().upper()
        if dev == "EUR":
            return normalize_number(row[eur_col])
        else:
            return normalize_number(row[huf_col])
    except:
        return None




def merge_with_minta(df_extracted, df_minta, invoice_col_extracted="Számlaszám", invoice_col_minta="Bizonylatszám"):
    df_merged = pd.merge(df_minta, df_extracted, how='outer', left_on=invoice_col_minta, right_on=invoice_col_extracted)
    matched = df_merged[invoice_col_extracted].notna().sum()
    total = len(df_minta)
    unmatched = total - matched
    match_rate = round(100 * matched / total, 2)
    
    stats = {
        "Összes minta sor": total,
        "Találatok száma": matched,
        "Hiányzó találatok": unmatched,
        "Egyezési arány (%)": match_rate
    }

    return df_merged, stats



# Inicializáljuk a session state változókat
if 'extracted_text_from_invoice' not in st.session_state:
    st.session_state.extracted_text_from_invoice = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []
if 'df_extracted' not in st.session_state:
    st.session_state.df_extracted = pd.DataFrame()
if 'df_minta' not in st.session_state:
    st.session_state.df_minta = pd.DataFrame()
if 'df_nav' not in st.session_state:
    st.session_state.df_nav = pd.DataFrame()
if 'df_karton' not in st.session_state:
    st.session_state.df_karton = pd.DataFrame()
if 'df_merged' not in st.session_state:
    st.session_state.df_merged = pd.DataFrame()
if 'df_merged_full' not in st.session_state:
    st.session_state.df_merged_full = pd.DataFrame()
if 'number_of_tokens' not in st.session_state:
    st.session_state.number_of_tokens = 0

openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]

st.title("📄 Számlaadat-kinyerő alkalmazás")

col_pdf, col_excel = st.columns([1, 1])  # nagyobb bal oldali hasáb

with col_pdf:
    st.subheader("📂 PDF-ek kinyerése")
    
    st.write("1) Tölts fel egy vagy több **magyar nyelvű számlát (PDF)**, amelyekből a rendszer kiolvassa a legfontosabb adatokat.")
    
    # 0) Fájl feltöltő
    uploaded_files = st.file_uploader("📤 PDF fájlok feltöltése", type=["pdf"], accept_multiple_files=True)
    
    asd = """selected_model = st.selectbox(
        "Válassz modellt az adatkinyeréshez:",
        ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
        index=0,
        help="Árak per 1M token:\n"
             "- gpt-4o: \\$2.50 input / \\$10 output\n"
             "- gpt-4.1: \\$1.25 input / \\$10 output\n"
             "- gpt-4.1-mini: \\$0.25 input / \\$2 output\n"
             "- gpt-4.1-nano: \\$0.05 input / \\$0.40 output"
        )"""
    
    selected_model = "gpt-4o"
    
    # 1) PDF feldolgozás
    if st.button("📑 Adatkinyerés a PDF-ből"):  
        st.session_state.extracted_text_from_invoice = []      
        if uploaded_files:
            if len(uploaded_files) > 200:
                st.write("⚠️ Az első 200 fájl kerül feldolgozásra.")
            files_to_process = uploaded_files[:200]
    
            # ---- 1) PDF szöveg kinyerés progress bar ----
            pdf_progress = st.progress(0, text="PDF szöveg kinyerése folyamatban...")
            pdf_status = st.empty()  # helyfoglaló a státusznak
    
            for idx, uploaded_file in enumerate(files_to_process, start=1):
                file_name = uploaded_file.name
                pdf_status.write(f"{file_name} feldolgozása (PDF szöveg kinyerése)...")
                pdf_text = extract_text_from_pdf(uploaded_file)
    
                if pdf_text is None:
                    continue
    
                st.session_state.extracted_text_from_invoice.append([file_name, pdf_text])

    
                # update progress
                pdf_progress.progress(idx / len(files_to_process), 
                                      text=f"PDF kinyerés: {idx}/{len(files_to_process)} kész")
    
            pdf_progress.empty()
            pdf_status.write("✅ PDF szöveg kinyerés befejezve.")
    
        else:
            st.warning("⚠️ Kérlek, tölts fel legalább egy PDF fájlt.")
    
        # ---- 2) GPT adatkinyerés progress bar ----
        st.session_state.extracted_data = []
        if st.session_state.extracted_text_from_invoice:
            gpt_progress = st.progress(0, text="AI adatkinyerés folyamatban...")
            gpt_status = st.empty()  # helyfoglaló a státusznak
    
            for idx, (file_name, pdf_content) in enumerate(st.session_state.extracted_text_from_invoice, start=1):
                gpt_status.write(f"{file_name} feldolgozása (AI adatkinyerés)...")
                extracted_rows, tokens = extract_data_with_gpt(file_name, pdf_content, selected_model)
                if extracted_rows:
                    st.session_state.extracted_data.extend(extracted_rows)
                    st.session_state.number_of_tokens += tokens
    
                # update progress
                gpt_progress.progress(idx / len(st.session_state.extracted_text_from_invoice),
                                      text=f"AI feldolgozás: {idx}/{len(st.session_state.extracted_text_from_invoice)} kész")
    
            gpt_progress.empty()
            gpt_status.write("✅ AI adatkinyerés befejezve.")
    
        # ---- 3) DataFrame létrehozás ----
        if st.session_state.extracted_data:
            st.session_state.df_extracted = pd.DataFrame(
                st.session_state.extracted_data,
                columns=["Fájlnév", "Eladó", "Vevő", "Számlaszám", "Számla kelte", 
                         "Bruttó ár", "Nettó ár", "ÁFA", "Deviza", "Árfolyam", "Megjegyzések"]
            )
            st.session_state.df_extracted["Számlaszám"] = st.session_state.df_extracted["Számlaszám"].astype(str)


    
    if len(st.session_state.df_extracted) > 0:        
        st.write("✅ **Adatok kinyerve!** Az alábbi táblázat tartalmazza az eredményeket:")
        st.dataframe(make_arrow_compatible(st.session_state.df_extracted))
    
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df_extracted.to_excel(writer, sheet_name='Adatok', index=False)
    
        # 🔑 reset pointer
        buffer.seek(0)
    
        st.download_button(
            label="📥 Kinyert adatok letöltése Excelben",
            data=buffer,
            file_name='kinyert_adatok.xlsx',
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


    # Token ár becslés
    price_input = st.session_state.number_of_tokens * MODEL_PRICES[selected_model]["input"] / 1_000_000
    st.write(f"💰 A becsült feldolgozási költség eddig: **${price_input:.2f}** ({selected_model})")

        
    import requests
    import xml.etree.ElementTree as ET
    import datetime
    
    def get_mnb_eur_rate(date_str, max_lookback_days=7, debug=False):
        """
        Gets the EUR/HUF rate from MNB SOAP webservice for the given date.
        Falls back to earlier dates if rate not found.
        """
        soap_url = "http://www.mnb.hu/arfolyamok.asmx"
        soap_action = "http://www.mnb.hu/webservices/MNBArfolyamServiceSoap/GetExchangeRates"
        headers = {
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": soap_action,
        }
    
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
        for i in range(max_lookback_days + 1):
            check_date = (dt - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
    
            # SOAP request body
            soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
            <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                           xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                           xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
              <soap:Body>
                <GetExchangeRates xmlns="http://www.mnb.hu/webservices/">
                  <startDate>{check_date}</startDate>
                  <endDate>{check_date}</endDate>
                  <currencyNames>EUR</currencyNames>
                </GetExchangeRates>
              </soap:Body>
            </soap:Envelope>"""
    
            debug = False
            if debug:
                st.code(soap_body, language="xml")
    
            try:
                resp = requests.post(soap_url, headers=headers, data=soap_body, timeout=10)
                resp.raise_for_status()
    
                if debug:
                    st.text_area(f"SOAP response for {check_date}", resp.text[:2000], height=150)
    
                # Parse XML response
                root = ET.fromstring(resp.content)
                ns = {"mnb": "http://www.mnb.hu/webservices/"}
    
                rate_text = root.find(".//mnb:GetExchangeRatesResult", ns)
                if rate_text is None or not rate_text.text:
                    continue
    
                # inner XML (string)
                inner_xml = ET.fromstring(rate_text.text)
                rate = inner_xml.find(".//Rate")
                if rate is not None and rate.text:
                    return float(rate.text.replace(",", "."))
    
            except Exception as e:
                if debug:
                    st.warning(f"Attempt {i+1} ({check_date}) failed: {e}")
                continue
    
        st.warning(f"❌ Nem található MNB árfolyam {date_str} vagy az előző {max_lookback_days} napban.")
        return None

    rate = get_mnb_eur_rate("2024-10-15", debug=True)
    print(rate)

    
    # --- 💶 Árfolyam ellenőrzés blokk ---
    if not st.session_state.df_extracted.empty:
        df_fx_check = st.session_state.df_extracted.copy()
    
        # szűrés: csak EUR és Árfolyam != 1
        df_fx_check = df_fx_check[
            (df_fx_check["Deviza"].str.upper() == "EUR")
        ].copy()
    
        if not df_fx_check.empty:
            st.markdown("### 💶 Árfolyam ellenőrzés (MNB hivatalos árfolyam alapján)")
    
            # NEW: Debug toggle
            #debug_fx = st.checkbox("🔧 Show API request/response debug", value=False, help="Print the full URL and raw XML/HTML response for each query/attempt.")
    
            if st.button("🔍 Árfolyamok ellenőrzése"):
                mnb_rates = []
                rate_matches = []
    
                for _, row in df_fx_check.iterrows():
                    try:
                        date_str = str(row["Számla kelte"]).strip()
                        parsed_date = pd.to_datetime(date_str, errors="coerce")
                        if pd.isna(parsed_date):
                            mnb_rates.append(None)
                            rate_matches.append("Nincs adat")
                            continue
    
                        mnb_date = parsed_date.strftime("%Y-%m-%d")
                        # pass debug flag through
                        mnb_rate = get_mnb_eur_rate(mnb_date, debug=False)
                        mnb_rates.append(mnb_rate)
    
                        if mnb_rate is not None:
                            cmp = compare_with_tolerance(row["Árfolyam"], mnb_rate, tolerance=1)
                            rate_matches.append(cmp)
                        else:
                            rate_matches.append("Nincs adat")
    
                    except Exception:
                        mnb_rates.append(None)
                        rate_matches.append("Nincs adat")

    
                df_fx_check["MNB árfolyam"] = mnb_rates
                df_fx_check["Árfolyam egyezik?"] = [x.replace('Igen', '✅ Igen').replace('Nem', '❌ Nem') for x in rate_matches]
    
                # Statisztika
                total_fx = len(df_fx_check)
                matched_fx = sum(1 for x in rate_matches if x == "Igen")
                match_rate = round(100 * matched_fx / total_fx, 2) if total_fx else 0.0
    
                st.write(f"**Ellenőrzött EUR számlák száma:** {total_fx}")
                st.write(f"**Egyező árfolyamok:** {matched_fx}  ({match_rate}%)")
    
                st.dataframe(make_arrow_compatible(df_fx_check[[
                    "Fájlnév", "Számlaszám", "Számla kelte", "Deviza",
                    "Árfolyam", "MNB árfolyam", "Árfolyam egyezik?"
                ]]))
    
                # Excel letöltés opció
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_fx_check.to_excel(writer, sheet_name='Arfolyam_ellenorzes', index=False)
                buffer.seek(0)
    
                st.download_button(
                    label="📥 Árfolyam ellenőrzés letöltése Excelben",
                    data=buffer,
                    file_name='arfolyam_ellenorzes.xlsx',
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    


with col_excel:
    
    st.subheader("📂 Excel fájlok betöltése")
    
    
    # --- Excel fájlok betöltése ---
    st.markdown("1) Töltsd fel a **Mintavétel** Excel fájlt:")
    uploaded_excel_file_minta = st.file_uploader(
        "📤 Mintavétel Excel feltöltése",
        type=["xlsx"],
        accept_multiple_files=False,
        help="Az adatok az első munkalapon a 10. sortól induljanak, és legyen 'Bizonylatszám' nevű oszlop."
    )
    
    
    if uploaded_excel_file_minta:
        try:
            st.session_state.df_minta = pd.read_excel(uploaded_excel_file_minta, skiprows=range(1, 9))
            st.session_state.df_minta.columns = list(st.session_state.df_minta.iloc[0])
            st.session_state.df_minta = st.session_state.df_minta.iloc[1:]
            st.session_state.df_minta["Bizonylatszám"] = st.session_state.df_minta["Bizonylatszám"].astype(str)
        except:
            st.warning("❌ Nem sikerült beolvasni a Mintavétel fájlt.")
    
    if len(st.session_state.df_minta) > 0:        
        st.write("✅ **Mintavétel betöltve!** Első néhány sor:")
        st.dataframe(make_arrow_compatible(st.session_state.df_minta.head(5)))
    
    
    
    # NAV fájl
    st.markdown("2) Töltsd fel a **NAV** Excel fájlt:")
    uploaded_excel_file_nav = st.file_uploader(
        "📤 NAV Excel feltöltése",
        type=["xlsx"],
        accept_multiple_files=False,
        help="Az adatok az első munkalapon a 6. sortól induljanak, és legyen 'számlasorszám' nevű oszlop."
    )    
    if uploaded_excel_file_nav:
        try:
            st.session_state.df_nav = pd.read_excel(uploaded_excel_file_nav, skiprows=5)
            st.session_state.df_nav["számlasorszám"] = st.session_state.df_nav["számlasorszám"].astype(str)
        except:
            st.warning("❌ Nem sikerült beolvasni a NAV fájlt.")
    
    if len(st.session_state.df_nav) > 0:        
        st.write("✅ **NAV fájl betöltve!** Első néhány sor:")
        st.dataframe(make_arrow_compatible(st.session_state.df_nav.head(5)))
    
    # Karton fájl
    st.markdown("3) Töltsd fel a **Karton** Excel fájlt:")
    uploaded_excel_file_karton = st.file_uploader(
        "📤 Karton Excel feltöltése",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        help="Az adatok az első munkalapon az A1 cellától induljanak."
    )    
    
    if uploaded_excel_file_karton:
        try:
            # Simply read the Excel file, no special column assumptions
            st.session_state.df_karton = pd.read_excel(uploaded_excel_file_karton)
    
            st.write("✅ **Karton betöltve!** Első néhány sor:")
            st.dataframe(make_arrow_compatible(st.session_state.df_karton.head(5)))
    
        except Exception as e:
            st.warning(f"❌ Nem sikerült beolvasni a Karton fájlt: {e}")
    
     
st.title("📄 Ellenőrzések")

col_left, col_right = st.columns([1, 1])  # nagyobb bal oldali hasáb

with col_left:
    st.subheader("📎 Kinyert adatok összefűzése és ellenőrzése: Mintavétel")
    
    invoice_colname_minta = "Bizonylatszám"
        
    if st.button("🔗 Összefűzés és ellenőrzés a Mintavétel excellel"):
        # Előfeltételek
        missing = []
        if not df_ready(st.session_state.df_extracted, required_cols=["Számlaszám", "Nettó ár"]):
            missing.append("Kinyert adatok (PDF feldolgozás)")
        if not df_ready(st.session_state.df_minta, required_cols=["Bizonylatszám", "Érték", "Érték deviza", "Devizanem"]):
            missing.append("Mintavétel Excel")
    
        if missing:
            need_msg(missing)
        else:
            try:
                df_minta = st.session_state.df_minta.copy()
                # Fejlécek biztonságos tisztítása (nem használ .str-t)
                df_minta.columns = [str(c).strip() for c in df_minta.columns]
                df_minta["Bizonylatszám"] = df_minta["Bizonylatszám"].astype(str)
    
                df_gpt = st.session_state.df_extracted.copy()
                df_gpt["Számlaszám"] = df_gpt["Számlaszám"].astype(str)
                    
                # --- átnevezés suffix-szel ---
                df_gpt = df_gpt.add_suffix("_ai")
                df_minta = df_minta.add_suffix("_minta")
                
                # ⬅️ teljes outer merge – hogy a csak-AI sorok is bekerüljenek
                df_merged_minta = pd.merge(
                    df_minta,
                    df_gpt,
                    how="outer",
                    left_on="Bizonylatszám_minta",
                    right_on="Számlaszám_ai",
                    indicator=True
                )
                
                # Találati státusz
                status_map = {
                    "both": "🔗 Egyezés",
                    "left_only": "📄 Csak Mintavétel",
                    "right_only": "🤖 Csak AI (PDF)"
                }
                df_merged_minta["Találat státusz"] = df_merged_minta["_merge"].map(status_map)
                
                # Nettó összehasonlítás
                df_merged_minta["Nettó egyezik?"] = df_merged_minta.apply(
                    lambda row: compare_with_tolerance(
                        get_minta_amount(
                            row,
                            huf_col="Érték_minta",
                            eur_col="Érték deviza_minta",
                            currency_col="Devizanem_minta"
                        ),
                        normalize_number(row.get("Nettó ár_ai")),
                        tolerance=5
                    ),
                    axis=1
                )
                
                # Minden egyezik?
                df_merged_minta["Minden egyezik?"] = df_merged_minta["Nettó egyezik?"].map({
                    "Igen": "✅ Igen",
                    "Nem": "❌ Nem",
                    "Nincs adat": ""
                })
                
                # --- RENDEZÉS: ---
                
                # cél: felül az egyezések (both), középen a csak minták (left_only), alul a csak AI (right_only)
                status_order = pd.CategoricalDtype(categories=["both", "left_only", "right_only"], ordered=True)
                df_merged_minta["_merge"] = df_merged_minta["_merge"].astype(status_order)
                
                # jelöljük, ha teljesen egyezik a nettó érték
                df_merged_minta["__ok_order"] = df_merged_minta["Minden egyezik?"].eq("✅ Igen").astype(int)
                
                # 🔧 most a kategória sorrend garantálja a helyes sorrendet
                df_merged_minta = (
                    df_merged_minta
                    .sort_values(
                        by=["_merge", "__ok_order"],
                        ascending=[True, False]  # Egyezés felül, "✅ Igen" előrébb
                    )
                    .drop(columns=["_merge", "__ok_order"])
                    .reset_index(drop=True)
                )

                
                st.session_state.df_merged_minta = df_merged_minta
                
                # --- statisztika ---
                total = len(df_merged_minta)
                matched = (df_merged_minta["Minden egyezik?"] == "✅ Igen").sum()
                match_rate = round(100 * matched / total, 2) if total else 0.0
                
                st.session_state.stats_minta = {
                    "Összes sor a táblában": total,
                    "Minden egyezés": matched,
                    "Egyezési arány (%)": match_rate
                }
                
                st.success("✅ Összefűzés és rendezés kész — az egyezések felül, csak minták középen, csak AI alul!")

    
            except Exception as e:
                st.error(f"Váratlan hiba történt a Mintavétel összefűzés során: {e}")

    
    if "df_merged_minta" in st.session_state:
        st.write("📄 **Összefűzött és ellenőrzött táblázat – Mintavétel:**")
        st.dataframe(make_arrow_compatible(st.session_state.df_merged_minta))
    
        csv_minta = st.session_state.df_merged_minta.to_csv(index=False).encode("utf-8")
    
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df_merged_minta.to_excel(writer, sheet_name='Minta', index=False)
    
        # 🔑 Reset buffer pointer to the start
        buffer.seek(0)
    
        st.download_button(
            label="📥 Letöltés Excel (Mintavétel)",
            data=buffer,
            file_name='merged_minta.xlsx',
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
        st.markdown("### 📊 Statisztika – Mintavétel ellenőrzés")
        for k, v in st.session_state.stats_minta.items():
            st.write(f"**{k}:** {v}")


with col_right:
    st.subheader("📎 Kinyert adatok összefűzése és ellenőrzése: NAV")
    
    if st.button("🔗 Összefűzés és ellenőrzés a NAV excellel"):
        # Előfeltételek
        missing = []
        if not df_ready(st.session_state.df_extracted, required_cols=["Számlaszám", "Bruttó ár", "Nettó ár", "ÁFA"]):
            missing.append("Kinyert adatok (PDF feldolgozás)")
        nav_required = ["számlasorszám"]
        if not df_ready(st.session_state.df_nav, required_cols=nav_required):
            missing.append("NAV Excel")
    
        if missing:
            need_msg(missing)
        else:
            try:
                # NAV adat előkészítés
                df_nav = st.session_state.df_nav.copy()
                df_nav.columns = [str(c).strip() for c in df_nav.columns]
                df_nav["számlasorszám"] = df_nav["számlasorszám"].astype(str)
                
                # GPT adat előkészítés
                df_gpt = st.session_state.df_extracted.copy()
                df_gpt["Számlaszám"] = df_gpt["Számlaszám"].astype(str)
                
                # --- átnevezés suffix-szel ---
                df_gpt = df_gpt.add_suffix("_ai")
                df_nav = df_nav.add_suffix("_nav")
                
                # NAV aggregálás számlaszám szinten
                agg_dict = {}
                for col in ["bruttó érték_nav", "bruttó érték Ft_nav", "nettóérték_nav", "nettóérték Ft_nav", "adóérték_nav", "adóérték Ft_nav"]:
                    if col in df_nav.columns:
                        agg_dict[col] = "sum"
                df_nav_sum = df_nav.groupby("számlasorszám_nav", as_index=False).agg(agg_dict)
                
                # Összefűzés számlaszám szintű összehasonlításhoz
                df_check = pd.merge(
                    df_gpt,
                    df_nav_sum,
                    how="left",
                    left_on="Számlaszám_ai",
                    right_on="számlasorszám_nav"
                )
                
                # Oszlopnevek rugalmas keresése
                brutto_col = next((c for c in ["bruttó érték_nav", "bruttó érték Ft_nav"] if c in df_check.columns), None)
                netto_col  = next((c for c in ["nettóérték_nav", "nettóérték Ft_nav"] if c in df_check.columns), None)
                afa_col    = next((c for c in ["adóérték_nav", "adóérték Ft_nav"] if c in df_check.columns), None)
                
                # Összegellenőrzések
                df_check["Bruttó egyezik?"] = df_check.apply(
                    lambda row: compare_with_tolerance(
                        normalize_number(row.get(brutto_col)) if brutto_col else None,
                        normalize_number(row.get("Bruttó ár_ai")),
                    ),
                    axis=1
                )
                df_check["Nettó egyezik?"] = df_check.apply(
                    lambda row: compare_with_tolerance(
                        normalize_number(row.get(netto_col)) if netto_col else None,
                        normalize_number(row.get("Nettó ár_ai")),
                    ),
                    axis=1
                )
                df_check["ÁFA egyezik?"] = df_check.apply(
                    lambda row: compare_with_tolerance(
                        normalize_number(row.get(afa_col)) if afa_col else None,
                        normalize_number(row.get("ÁFA_ai")),
                    ),
                    axis=1
                )
                
                # Minden egyezik? logika:
                # - Ha mindhárom "Igen" → ✅ Igen
                # - Ha bármelyik "Nem" → ❌ Nem
                # - Egyébként (legalább egy "Nincs adat", de nincs "Nem") → Nincs adat
                def overall_match(row):
                    results = [row["Bruttó egyezik?"], row["Nettó egyezik?"], row["ÁFA egyezik?"]]
                    if all(r == "Igen" for r in results):
                        return "✅ Igen"
                    elif any(r == "Nem" for r in results):
                        return "❌ Nem"
                    else:
                        return "Nincs adat"
                
                df_check["Minden egyezik?"] = df_check.apply(overall_match, axis=1)
                
                # --- Részletező tábla ---
                df_details = pd.merge(
                    df_nav,   # NAV oszlopok _nav
                    df_gpt,   # GPT oszlopok _ai
                    how="right",
                    left_on="számlasorszám_nav",
                    right_on="Számlaszám_ai"
                )
                
                # Számlaszintű ellenőrzések visszacsatolása
                df_details = pd.merge(
                    df_details,
                    df_check[["Számlaszám_ai", "Bruttó egyezik?", "Nettó egyezik?", "ÁFA egyezik?", "Minden egyezik?"]],
                    how="left",
                    on="Számlaszám_ai"
                )
                
                # Mentés session_state-be
                st.session_state.df_merged_nav = df_details
                
                # Statisztika számlaszám szinten
                total = len(df_check)
                matched_all = (df_check["Minden egyezik?"] == "✅ Igen").sum()
                match_rate = round(100 * matched_all / total, 2)
                
                st.session_state.stats_nav = {
                    "Összes számla": total,
                    "Minden egyezés": matched_all,
                    "Teljes egyezési arány (%)": match_rate
                }
                
                st.success("✅ NAV fájllal való összefűzés és ellenőrzés kész!")

            except Exception as e:
                st.error(f"Váratlan hiba történt a NAV összefűzés során: {e}")

    if "df_merged_nav" in st.session_state:
        st.write("📄 **Összefűzött és ellenőrzött táblázat – NAV (tételszinten):**")
        st.dataframe(make_arrow_compatible(st.session_state.df_merged_nav))

        # Excel letöltés
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df_merged_nav.to_excel(writer, sheet_name='NAV részletek', index=False)

        st.download_button(
            label="📥 Letöltés Excel (NAV részletek)",
            data=buffer,
            file_name="merged_nav.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("### 📊 Statisztika – NAV összehasonlítás")
        for k, v in st.session_state.stats_nav.items():
            st.write(f"**{k}:** {v}")


col_karton, col_mintanav = st.columns([1, 1])


with col_karton:
    st.subheader("📎 Kinyert adatok összefűzése: Karton")
    
    if st.button("🔗 Összefűzés a Kartonnal"):
        try:
            # GPT számlaszámok
            invoice_numbers = st.session_state.df_extracted["Számlaszám"].astype(str).unique()
    
            # Karton tábla előkészítése
            df_karton = st.session_state.df_karton.copy()
            df_karton.columns = [str(c).strip() for c in df_karton.columns]
    
            # Szűrés: minden olyan sor kell, ahol bármelyik oszlopban szerepel a számlaszám
            mask = df_karton.apply(lambda row: row.astype(str).isin(invoice_numbers).any(), axis=1)
            df_filtered_karton = df_karton[mask].copy()
    
            # Ha van "Bizonylat" vagy "Számlaszám" oszlop, rendezzük arra
            for possible_col in ["Bizonylat", "Számlaszám", "számlasorszám"]:
                if possible_col in df_filtered_karton.columns:
                    df_filtered_karton = df_filtered_karton.sort_values(by=possible_col)
                    break
    
            st.session_state.df_filtered_karton = df_filtered_karton
    
            # Statisztika: hány GPT számlaszámhoz találtunk sorokat
            matched_karton = df_filtered_karton.apply(
                lambda row: any(str(val) in invoice_numbers for val in row.values), axis=1
            ).sum()
            total_karton = len(invoice_numbers)
    
            st.session_state.stats_karton = {
                "Összes számla (GPT)": total_karton,
                "Kartonban megtalált sorok": matched_karton,
            }
    
            st.success("✅ Karton keresés és szűrés kész!")
    
        except Exception as e:
            st.error(f"❌ Hiba történt a Karton keresés során: {e}")
    
    if "df_filtered_karton" in st.session_state:
        st.write("📄 **Szűrt táblázat – Karton (csak releváns sorok):**")
        st.dataframe(make_arrow_compatible(st.session_state.df_filtered_karton))
    
        # Excel letöltés előkészítés
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            st.session_state.df_filtered_karton.to_excel(writer, sheet_name="Karton szűrt", index=False)
    
        st.download_button(
            label="📥 Letöltés Excel (Karton)",
            data=buffer,
            file_name="filtered_karton.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
        st.markdown("### 📊 Statisztika – Karton keresés")
        for k, v in st.session_state.stats_karton.items():
            st.write(f"**{k}:** {v}")
    
with col_mintanav:
    st.subheader("📎 Minta és NAV adatok összefűzése")

    if st.button("🔗 Összefűzés a Minta és NAV adatok között"):
        missing = []
        if not df_ready(st.session_state.df_minta, required_cols=["Bizonylatszám"]):
            missing.append("Mintavétel Excel")
        if not df_ready(st.session_state.df_nav, required_cols=["számlasorszám"]):
            missing.append("NAV Excel")

        if missing:
            need_msg(missing)
        else:
            try:
                df_minta = st.session_state.df_minta.copy()
                df_nav = st.session_state.df_nav.copy()

                df_minta.columns = [str(c).strip() for c in df_minta.columns]
                df_nav.columns = [str(c).strip() for c in df_nav.columns]

                df_minta["Bizonylatszám"] = df_minta["Bizonylatszám"].astype(str)
                df_nav["számlasorszám"] = df_nav["számlasorszám"].astype(str)

                # add suffixes to avoid column name collisions
                df_minta_s = df_minta.add_suffix("_minta")
                df_nav_s = df_nav.add_suffix("_nav")

                # Left join (Minta as left)
                df_minta_nav = pd.merge(
                    df_minta_s,
                    df_nav_s,
                    how="left",
                    left_on="Bizonylatszám_minta",
                    right_on="számlasorszám_nav"
                )

                st.session_state.df_minta_nav = df_minta_nav

                # Simple stats
                total_rows = len(df_minta_nav)
                matched = df_minta_nav["számlasorszám_nav"].notna().sum()
                unmatched = total_rows - matched
                match_rate = round(100 * matched / total_rows, 2) if total_rows else 0.0

                st.session_state.stats_minta_nav = {
                    "Összes Minta sor": total_rows,
                    "NAV-ban talált számlák": matched,
                    "Hiányzó számlák": unmatched,
                    "Egyezési arány (%)": match_rate,
                }

                st.success("✅ Minta és NAV adatok összefűzése kész!")

            except Exception as e:
                st.error(f"❌ Hiba történt az összefűzés során: {e}")

    if "df_minta_nav" in st.session_state:
        st.write("📄 **Összefűzött táblázat – Minta × NAV:**")
        st.dataframe(make_arrow_compatible(st.session_state.df_minta_nav))

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            st.session_state.df_minta_nav.to_excel(writer, sheet_name="Minta_NAV", index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Letöltés Excel (Minta–NAV)",
            data=buffer,
            file_name="merged_minta_nav.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("### 📊 Statisztika – Minta × NAV összefűzés")
        for k, v in st.session_state.stats_minta_nav.items():
            st.write(f"**{k}:** {v}")




local_test = r"""

os.listdir('./')


import tomllib

secrets_path = r"C:\Users\abele\.streamlit\secrets.toml"

with open(secrets_path, "rb") as f:
    secrets = tomllib.load(f)

openai.api_key = secrets["OPENAI_API_KEY"]

MODEL = "gpt-4o"  # cheapest useful model
PDF_FILE = "9601013661.pdf"  # <-- replace with your own file path


with open(PDF_FILE, "rb") as f:
    text = extract_text_from_pdf(f)


# 2) GPT extraction
rows, tokens_used = extract_data_with_gpt(PDF_FILE, text, MODEL)


# 3) Create DataFrame
df = pd.DataFrame(rows, columns=[
    "Fájl", "Eladó", "Vevő", "Számlaszám", "Számla dátum",
    "Bruttó ár", "Nettó ár", "ÁFA", "Pénznem", "Árfolyam"
])





"""



local_test = r"""
df_extracted = pd.read_csv('extract_data.csv')
df_minta = pd.read_excel('Mintavétel_költségek_Sonneveld és ellenïrzés.xlsx', sheet_name='Mintavétel', skiprows = range(1, 9))
df_minta.columns = list(df_minta.iloc[0])
df_minta = df_minta.iloc[1:]
df_minta["Bizonylatszám"] = df_minta["Bizonylatszám"].astype(str)
df_karton = pd.read_excel('Könyvelési karton 2024_Sonneveld Kft.xlsx', sheet_name='Munka1')

df_temp = pd.merge(df_minta, df_extracted, how='outer', left_on='Bizonylatszám', right_on='Számlaszám')
nr_of_columns = len(df_temp.columns)

df_temp = pd.merge(df_temp, df_karton, how='left', left_on='Bizonylatszám', right_on='Bizonylat')

column_to_compare = 'Bizonylatszám'
columns_to_delete = df_temp.columns[:nr_of_columns]
df_merged = replace_successive_duplicates(df_temp, column_to_compare, columns_to_delete)

"""
