import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime
import os

# Path to history file
HISTORY_FILE = "history/yoga_history.txt"
cnn_model = "model/efficientnetb0.keras"

# Load model
model = load_model(
    cnn_model,
    custom_objects={'preprocess_input': preprocess_input}
)

class_names = [
    'Поза стільця (Chair pose)', 'Собака обличчям вниз (Downward facing dog)',
    'Поза трикутника (Extended triangle pose)', 'Поза богині (Goddess pose)',
    'Планка (Plank)', 'Бокова планка (Side plank)',
    'Поза дерева (Tree pose)', 'Поза воїна 2 (Warrior 2 pose)'
]

# -------------------- HISTORY --------------------

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            return [line for line in lines if line.strip()]
    return []

def save_to_history(entry):
    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        f.write(entry + '\n')

def clear_plan():
    return ""

# -------------------- FRAME PROCESSING --------------------

def process_frame(frame):
    """
    Process a single video frame and classify yoga pose
    """
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img, verbose=0)
    confidence = float(np.max(preds))

    if confidence > 0.7:
        return class_names[np.argmax(preds)]

    return None

# -------------------- VIDEO PROCESSING --------------------

def process_video(video_path):
    """
    Process video and return a set of detected poses
    """
    cap = cv2.VideoCapture(video_path)
    detected_poses = set()

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # analyze every 5th frame
        if frame_counter % 5 == 0:
            pose = process_frame(frame)
            if pose:
                detected_poses.add(pose)

    cap.release()
    return detected_poses

# -------------------- MAIN FUNCTION --------------------

def analyze_video_with_plan(video_path, planned_text, add_to_history, history):

    if video_path is None:
        return "Upload a video for analysis.", history, "\n".join(history) or ""

    # 1. get video duration
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = frame_count / fps if fps > 0 else 0
    cap.release()

    # 2. pose recognition
    detected_poses = process_video(video_path)

    if not detected_poses:
        return "No poses detected or low confidence.", history, "\n".join(history) or ""

    # 3. process training plan
    planned_list = [p.strip() for p in planned_text.split('\n') if p.strip()]
    planned_set = set(planned_list)

    executed = planned_set.intersection(detected_poses)
    not_executed = planned_set - detected_poses

    result = (
        f"Executed from plan: {', '.join(sorted(executed)) or ''}\n"
        f"Not executed from plan: {', '.join(sorted(not_executed)) or ''}\n"
    )

    # 4. history handling
    if add_to_history:
        date_str = datetime.now().strftime("%Y-%m-%d")
        history_entry = (
            f"{date_str} | Executed poses: {', '.join(sorted(executed)) or ''} "
            f"| Duration: {total_duration:.0f} sec"
        )
        save_to_history(history_entry)
        history = history + [history_entry]

    history_display = "\n".join(history) if history else ""

    return result, history, history_display

# -------------------- GRADIO UI --------------------

with gr.Blocks() as demo:

    initial_history = load_history()
    history_state = gr.State(initial_history)

    with gr.Row():
        video_input = gr.Video(
            label="Upload video",
            sources=["upload"],
            format="mp4",
            interactive=True,
            height=400,
            width=500
        )

        with gr.Column():
            plan_text = gr.Textbox(
                label="Training plan",
                lines=6,
                interactive=False
            )

            plan_dropdown = gr.Dropdown(
                choices=class_names,
                label="Add yoga pose to plan",
                interactive=True
            )

            clear_btn = gr.Button("Clear")
            clear_btn.click(fn=clear_plan, outputs=plan_text)

    add_to_history_checkbox = gr.Checkbox(
        label="Add to history",
        value=False
    )

    btn = gr.Button("Analyze")

    output = gr.Textbox(label="Result", lines=8)

    history_output = gr.Textbox(
        label="History",
        value="\n".join(initial_history) or "",
        lines=10,
        interactive=False
    )

    def add_pose_to_plan(selected_pose, current_plan):
        if not selected_pose:
            return current_plan
        return (current_plan + "\n" + selected_pose).strip() if current_plan else selected_pose

    plan_dropdown.change(
        fn=add_pose_to_plan,
        inputs=[plan_dropdown, plan_text],
        outputs=plan_text
    )

    btn.click(
        fn=analyze_video_with_plan,
        inputs=[video_input, plan_text, add_to_history_checkbox, history_state],
        outputs=[output, history_state, history_output]
    )

demo.launch()
