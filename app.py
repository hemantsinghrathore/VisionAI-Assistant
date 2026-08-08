import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
from collections import Counter
import tempfile
import time
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="VisionAI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    """
    <style>

    .main-title {
        font-size: 52px;
        font-weight: 900;
        text-align: center;
        color: #312E81;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #475569;
        margin-bottom: 35px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(99, 102, 241, 0.18);
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }

    .stButton > button {
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.6rem;
        font-weight: 700;
    }

    .stDownloadButton > button {
        background: linear-gradient(90deg, #10B981, #22C55E);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.6rem;
        font-weight: 700;
    }
    </style>

    <div class="main-title">🤖 VisionAI Assistant</div>
    <div class="subtitle">
        An AI-powered computer vision app for object detection using YOLOv8
    </div>
    """,
    unsafe_allow_html=True
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.30,
    0.05
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
if "history" not in st.session_state:
    st.session_state.history = []
if "total_images" not in st.session_state:
    st.session_state.total_images = 0

if "all_detected_objects" not in st.session_state:
    st.session_state.all_detected_objects = []

input_mode = st.radio("Choose input mode:", ["Upload Image", "Use Camera"])

image = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif input_mode == "Use Camera":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file)

if image is not None:
    left, right = st.columns(2)

    with left:
     st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Objects"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            image_path = temp_file.name

        with st.spinner("Analyzing image with AI..."):
            start_time = time.time()
            results = model(image_path, conf=confidence_threshold)
            end_time = time.time()

        detection_time = end_time - start_time
        result_image = results[0].plot()

        with right:
         st.image(result_image, caption="Detected Objects", use_container_width=True)
        result_pil = Image.fromarray(result_image)
        img_buffer = io.BytesIO()
        result_pil.save(img_buffer, format="PNG")

        st.download_button(
            label="Download Detected Image",
            data=img_buffer.getvalue(),
            file_name="detected_objects.png",
            mime="image/png"
        )

        detected_objects = []
        detection_data = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            confidence = float(box.conf[0])

            detected_objects.append(object_name)
            detection_data.append({
                "Object": object_name,
                "Confidence": f"{confidence:.2f}"
            })

        if detected_objects:
            counts = Counter(detected_objects)
            summary = ", ".join([f"{count} {obj}" for obj, count in counts.items()])
            st.success(f"I can see: {summary}")
            
            average_confidence = sum(float(item["Confidence"]) for item in detection_data) / len(detection_data)
            st.info(
             f"🤖 AI Summary:\n\n"
             f"I detected **{summary}** in this image. "
             f"The average confidence is **{average_confidence*100:.1f}%**."
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Objects Detected", len(detected_objects))
            col2.metric("Average Confidence", f"{average_confidence:.2f}")
            col3.metric("Detection Time", f"{detection_time:.2f}s")

            st.subheader("Detection Details")

            for item in detection_data:
                st.write(f"**{item['Object']}**")
                st.progress(float(item["Confidence"]))
                st.write(f"Confidence: {float(item['Confidence'])*100:.1f}%")
                st.divider()
            st.table(detection_data)
            report_data = []

            for item in detection_data:
                report_data.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "object": item["Object"],
                    "confidence": item["Confidence"],
                    "detection_time_seconds": round(detection_time, 2)
                })

            report_df = pd.DataFrame(report_data)

            csv_report = report_df.to_csv(index=False)

            st.download_button(
                label="Download Detection Report",
                data=csv_report,
                file_name="visionai_detection_report.csv",
                mime="text/csv"
            )
            st.subheader("📊 Detection Analytics")

            history_entry = {
                "Objects": summary,
                "Count": len(detected_objects),
                "Average Confidence": round(average_confidence * 100, 1),
                "Time (s)": round(detection_time, 2)
            }

            st.session_state.history.append(history_entry)
            st.session_state.total_images += 1
            st.session_state.all_detected_objects.extend(detected_objects)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    names=list(counts.keys()),
                    values=list(counts.values()),
                    title="Detected Objects Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                confidence_chart = px.bar(
                    detection_data,
                    x="Object",
                    y="Confidence",
                    title="Confidence by Object"
                )
                st.plotly_chart(confidence_chart, use_container_width=True)

            st.subheader("🕒 Detection History")
            st.dataframe(st.session_state.history, use_container_width=True)
            st.subheader("Session Summary")

            session_counts = Counter(st.session_state.all_detected_objects)

            most_common_object = (
                session_counts.most_common(1)[0][0]
                if session_counts
                else "None"
            )

            s1, s2, s3 = st.columns(3)

            s1.metric("Images Processed", st.session_state.total_images)
            s2.metric("Total Objects Detected", len(st.session_state.all_detected_objects))
            s3.metric("Most Common Object", most_common_object)
        else:
            st.warning("No objects detected.")
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray; font-size: 14px;'>
                VisionAI Assistant | Built with Streamlit, Python and YOLOv8
            </div>
            """,
            unsafe_allow_html=True
        )