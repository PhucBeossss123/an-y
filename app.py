"""Curated Streamlit report for iSarcasmEval English Subtask A."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
TEAM_RESULTS = ROOT / "results" / "team"
PHUC_RESULTS = ROOT / "results" / "phuc" / "shared_seed42_20261017"
TEAM_OUTPUTS = ROOT / "outputs" / "team"
PHUC_OUTPUTS = ROOT / "outputs" / "shared_review_20261017"
MODEL_LABELS = {"bert": "BERT-base", "roberta": "RoBERTa-base", "distilbert": "DistilBERT"}
SEEDS = (42, 123, 2026)

st.set_page_config(page_title="iSarcasmEval | Báo cáo", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:4rem}
.hero{padding:1.5rem 1.7rem;border-radius:18px;background:linear-gradient(120deg,#102b46,#1c5b70);color:#fff;margin-bottom:1.2rem}
.hero h1{margin:0 0 .4rem;font-size:2rem}.hero p{margin:.25rem 0;color:#e0eef4}
.note{padding:.85rem 1rem;border-left:4px solid #287b86;background:#f1f7f8;border-radius:6px;margin:.8rem 0 1.1rem}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def model_table():
    return pd.read_csv(TEAM_RESULTS / "model_comparison.csv")


@st.cache_data
def phuc_summary():
    return pd.read_csv(PHUC_RESULTS / "summary.csv")


@st.cache_data
def prompt_table():
    return pd.read_csv(PHUC_RESULTS / "prompt_comparison.csv")


@st.cache_data
def selected_model():
    return json.loads((TEAM_OUTPUTS / "selected_model.json").read_text(encoding="utf-8"))


def pct(value):
    return f"{float(value):.1%}"


def score_cell(mean, sd):
    return f"{float(mean):.3f} ± {float(sd):.3f}"


def report_page():
    data = model_table().set_index("architecture")
    winner_key = selected_model()["architecture"]
    winner = data.loc[winner_key]
    svm = phuc_summary().query("method == 'tfidf_linear_svm' and split == 'official_test'").iloc[0]
    zero = phuc_summary().query("method == 'zero_shot' and prompt_id == 'explicit_negation' and split == 'official_test'").iloc[0]
    st.markdown(f"<section class='hero'><h1>Báo cáo phân loại mỉa mai</h1><p>iSarcasmEval English Subtask A · Nhãn 0 = sincere, 1 = sarcastic</p><p>Mô hình được chọn theo validation: <b>{MODEL_LABELS[winner_key]}, seed {int(selected_model()['seed'])}</b></p></section>", unsafe_allow_html=True)
    st.write("**Kết luận ngắn:** RoBERTa-base đạt Macro-F1 validation trung bình cao nhất trong ba encoder. Đây là mô hình được chọn trước khi xem official test.")
    st.markdown("<div class='note'><b>Cách đọc:</b> Macro-F1 cân bằng hai lớp; F1 sarcastic chỉ đo lớp mỉa mai. BERT/RoBERTa/DistilBERT là trung bình ± độ lệch chuẩn mẫu qua 3 seed. SVM và zero-shot là một lần chạy.</div>", unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Macro-F1 · validation",pct(winner.validation_macro_f1_mean),help="Trung bình qua ba seed; dùng để chọn kiến trúc.")
    b.metric("F1 sarcastic · validation",pct(winner.validation_f1_sarcastic_mean))
    c.metric("Macro-F1 · official test",pct(winner.test_macro_f1_mean))
    d.metric("F1 sarcastic · official test",pct(winner.test_f1_sarcastic_mean))
    st.subheader("So sánh trên official test")
    rows=[]
    for key,row in data.iterrows():
        rows.append({"Mô hình":MODEL_LABELS[key],"Macro-F1":score_cell(row.test_macro_f1_mean,row.test_macro_f1_std),"F1 sarcastic":score_cell(row.test_f1_sarcastic_mean,row.test_f1_sarcastic_std),"Lượt chạy":"3 seed"})
    rows += [
        {"Mô hình":"TF-IDF + SVM","Macro-F1":pct(svm.macro_f1),"F1 sarcastic":pct(svm.f1_sarcastic),"Lượt chạy":"1 lần"},
        {"Mô hình":"Zero-shot · prompt đã chọn","Macro-F1":pct(zero.macro_f1),"F1 sarcastic":pct(zero.f1_sarcastic),"Lượt chạy":"1 lần"},
    ]
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    st.caption("Các số test là kết quả báo cáo cuối sau khi cấu hình được chọn trên validation. Không dùng test để chọn mô hình.")
    chart=data[["validation_macro_f1_mean","test_macro_f1_mean"]].rename(columns={"validation_macro_f1_mean":"Validation","test_macro_f1_mean":"Official test"}).rename(index=MODEL_LABELS)
    st.bar_chart(chart)
    st.subheader("Điểm cần lưu ý khi phân tích")
    st.write("RoBERTa dẫn đầu Macro-F1 test trong các encoder; SVM đạt F1 sarcastic cao hơn RoBERTa trong các kết quả đơn lẻ này. Vì vậy nên nêu cả hai chỉ số, không chỉ xếp hạng theo một điểm.")
    st.write("Zero-shot với prompt được chọn có Macro-F1 test 44.8%, nhưng F1 sarcastic chỉ 4.3%. Prompt đó được chọn theo Macro-F1 validation; mức giảm của F1 sarcastic trên test là một hạn chế đáng báo cáo.")


def model_page():
    data=model_table().copy().set_index("architecture").rename(index=MODEL_LABELS)
    st.title("So sánh mô hình")
    st.write("Ba encoder được chạy với seed 42, 123, 2026. Kiến trúc được chọn theo Macro-F1 validation trung bình; từng kiến trúc chọn seed theo validation. Official test không tham gia chọn.")
    view=pd.DataFrame({
        "Validation · Macro-F1":data.apply(lambda r:score_cell(r.validation_macro_f1_mean,r.validation_macro_f1_std),axis=1),
        "Validation · F1 sarcastic":data.apply(lambda r:score_cell(r.validation_f1_sarcastic_mean,r.validation_f1_sarcastic_std),axis=1),
        "Official test · Macro-F1":data.apply(lambda r:score_cell(r.test_macro_f1_mean,r.test_macro_f1_std),axis=1),
        "Official test · F1 sarcastic":data.apply(lambda r:score_cell(r.test_f1_sarcastic_mean,r.test_f1_sarcastic_std),axis=1),
        "Seed tốt nhất theo validation":data.best_validation_seed.astype(int).astype(str),
    })
    st.dataframe(view,width="stretch")
    st.bar_chart(data[["validation_macro_f1_mean","test_macro_f1_mean"]].rename(columns={"validation_macro_f1_mean":"Validation Macro-F1","test_macro_f1_mean":"Test Macro-F1"}))
    st.info("± là độ lệch chuẩn mẫu giữa ba seed, không phải khoảng tin cậy. So sánh với SVM/zero-shot ở trang Báo cáo; các phương pháp đó chỉ có một lần chạy.")


def prompt_page():
    p=prompt_table().query("split == 'shared_validation_seed_42'").copy()
    names={"original":"Gốc","statement":"Câu phát biểu","explicit_negation":"Phủ định tường minh · được chọn","social_post":"Bài đăng mạng xã hội","ironic_or_literal":"Mỉa mai hoặc nghĩa đen"}
    p["Prompt"]=p.prompt_id.map(names)
    p=p.sort_values("macro_f1",ascending=False)
    st.title("Zero-shot: prompt nào được chọn?")
    st.write("Chỉ so prompt trên shared validation. Prompt có Macro-F1 validation cao nhất được chọn trước khi chạy official test.")
    st.dataframe(p[["Prompt","macro_f1","f1_sarcastic","precision_sarcastic","recall_sarcastic"]].rename(columns={"macro_f1":"Macro-F1","f1_sarcastic":"F1 sarcastic","precision_sarcastic":"Precision sarcastic","recall_sarcastic":"Recall sarcastic"}).style.format({"Macro-F1":"{:.3f}","F1 sarcastic":"{:.3f}","Precision sarcastic":"{:.3f}","Recall sarcastic":"{:.3f}"}),hide_index=True,width="stretch")
    st.bar_chart(p.set_index("Prompt")[["macro_f1","f1_sarcastic"]].rename(columns={"macro_f1":"Macro-F1","f1_sarcastic":"F1 sarcastic"}))
    s=phuc_summary().query("method == 'zero_shot' and split == 'official_test'").set_index("prompt_id")
    chosen=s.loc["explicit_negation"]; original=s.loc["original"]
    st.subheader("Official test: prompt đã chọn so với prompt gốc")
    x,y=st.columns(2)
    x.metric("Macro-F1 · prompt đã chọn",pct(chosen.macro_f1),f"{chosen.macro_f1-original.macro_f1:+.3f} so với prompt gốc")
    y.metric("F1 sarcastic · prompt đã chọn",pct(chosen.f1_sarcastic),f"{chosen.f1_sarcastic-original.f1_sarcastic:+.3f} so với prompt gốc",delta_color="normal")
    st.warning("Prompt được chọn cải thiện Macro-F1 test nhẹ so với prompt gốc nhưng F1 sarcastic thấp hơn. Đây là tradeoff cần nêu khi trình bày kết quả.")
    st.caption("Zero-shot NLI scores không phải xác suất đã hiệu chỉnh.")


@st.cache_data
def error_examples():
    from an_y.data import load_test
    test=load_test(ROOT/"data"/"raw").set_index("sample_id")
    ids=["test:127","test:176","test:246","test:281","test:10"]
    pred={}
    for key,seed in (("bert",42),("roberta",42),("distilbert",123)):
        f=pd.read_csv(TEAM_RESULTS/key/f"seed_{seed}"/"official_test_predictions.csv",dtype={"sample_id":str}).set_index("sample_id")
        pred[key]=f["y_pred"]
    reasons={"test:127":"Phủ định cuối câu","test:176":"Khen ngược / đối lập","test:246":"Phủ định nhấn mạnh","test:281":"Cấu trúc ‘not like … or anything’","test:10":"Emoji và từ ngữ tích cực"}
    out=[]
    for sid in ids:
        row=test.loc[sid]
        out.append({"ID":sid,"Nhóm":reasons[sid],"Câu":row.text,"Gold":int(row.label),**{MODEL_LABELS[k]:int(pred[k].loc[sid]) for k in pred}})
    return pd.DataFrame(out)


def errors_page():
    st.title("Một vài lỗi đáng phân tích")
    st.write("Các ví dụ dưới đây được chọn để minh họa nhóm lỗi đã ghi nhận, không phải toàn bộ dự đoán. Nhãn 0 = sincere, 1 = sarcastic.")
    try:
        examples=error_examples()
    except Exception as e:
        st.error(f"Chưa nạp được ví dụ lỗi từ official test local: {e}")
        return
    for _,r in examples.iterrows():
        with st.container(border=True):
            st.markdown(f"**{r['Nhóm']} · {r['ID']} · Gold: {r['Gold']}**")
            st.write(r["Câu"])
            st.caption(" · ".join(f"{m}: {r[m]}" for m in MODEL_LABELS.values()))
    st.caption("Phân tích định tính; không suy rộng tần suất từ các ví dụ được chọn.")


def demo_page():
    st.title("Thử phân loại một câu")
    chosen=selected_model(); default=MODEL_LABELS[chosen["architecture"]]
    choices=["RoBERTa-base","BERT-base","DistilBERT","TF-IDF + SVM","Zero-shot"]
    model=st.selectbox("Mô hình",choices,index=choices.index(default))
    text=st.text_area("Câu tiếng Anh",placeholder="Type an English sentence…",max_chars=5000)
    st.caption("Câu nhập tay không thuộc validation hay official test. Điểm/margin hiển thị không phải xác suất đã hiệu chỉnh.")
    if st.button("Phân loại",type="primary",disabled=not text.strip()):
        with st.spinner("Đang chạy mô hình…"):
            try:
                from an_y.predict import predict_baseline,predict_zero_shot
                from an_y.bert import predict_bert
                import json as _json
                if model=="TF-IDF + SVM": result=predict_baseline(text.strip(),PHUC_OUTPUTS)
                elif model=="Zero-shot": result=predict_zero_shot(text.strip(),"selected",PHUC_OUTPUTS)
                else:
                    key={v:k for k,v in MODEL_LABELS.items()}[model]
                    checkpoint=Path(_json.loads((TEAM_OUTPUTS/key/"selected_checkpoint.json").read_text(encoding="utf-8"))["checkpoint"])
                    result=predict_bert(text.strip(),checkpoint)
                label=int(result["y_pred"])
                st.success(f"Kết quả: {'sarcastic (1)' if label else 'sincere (0)'}")
                st.write(f"Mô hình: {model}")
                score=result.get("decision_margin",result.get("selected_score"))
                if score is not None: st.write(f"Điểm thô: {float(score):.4f} · {result.get('score_note','')}")
            except Exception as e:
                st.error(f"Không chạy được mô hình: {e}")


PAGES={"Báo cáo":report_page,"So sánh mô hình":model_page,"Prompt zero-shot":prompt_page,"Phân tích lỗi":errors_page,"Thử câu":demo_page}
page=st.sidebar.radio("Nội dung",list(PAGES),label_visibility="visible")
PAGES[page]()
