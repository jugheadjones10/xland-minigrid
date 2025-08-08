import os

import gradio as gr

# VIDEO_ROOT = "/Users/kimyoungjin/Projects/monkey/xland-minigrid/experiments_complete/rl2-level0-all-4000train-400test-488steps-videos/pushworld_videos"
VIDEO_ROOT = "pushworld_videos"

MAX_VIDEOS_DISPLAY = 50
VIDEOS_PER_ROW = 5


# --- Main Application Logic ---
def get_videos_for_grid(dataset, status):
    """
    Retrieves video file paths and creates a list of Gradio updates
    to populate the grid of gr.Video components.
    """
    # Sanitize inputs to create the directory path
    dataset_folder = dataset.lower()
    status_folder = status.lower()
    video_dir = os.path.join(VIDEO_ROOT, dataset_folder, status_folder)

    print(f"Searching for videos in: {video_dir}")

    video_files = []
    if os.path.isdir(video_dir):
        try:
            supported_formats = (".mp4", ".webm", ".ogg")
            all_files = sorted(os.listdir(video_dir))  # Sort for consistent order
            video_files = [os.path.join(video_dir, f) for f in all_files if f.lower().endswith(supported_formats)]
            print(f"Found {len(video_files)} videos.")
        except Exception as e:
            print(f"An error occurred while accessing files: {e}")
            video_files = []

    # --- Create the list of updates for the video components ---
    updates = []
    # We only display up to MAX_VIDEOS_DISPLAY
    videos_to_show = video_files[:MAX_VIDEOS_DISPLAY]

    for i in range(MAX_VIDEOS_DISPLAY):
        if i < len(videos_to_show):
            # If there is a video for this slot, update it with the path and make it visible.
            updates.append(gr.update(value=videos_to_show[i], visible=True))
        else:
            # Otherwise, clear the video and hide the component.
            updates.append(gr.update(value=None, visible=False))

    return updates


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # PushWorld Level 0 All RL^2 Videos
        """
    )

    with gr.Row():
        # Input controls for filtering
        dataset_filter = gr.Radio(["Train", "Test"], label="Dataset", value="Train", info="Select the dataset to view.")
        status_filter = gr.Radio(
            ["Succeeded", "Failed"], label="Outcome", value="Succeeded", info="Filter by puzzle outcome."
        )

    # --- Create a dynamic grid of Video components ---
    video_outputs = []
    with gr.Column():
        # Create rows for the grid
        for i in range(0, MAX_VIDEOS_DISPLAY, VIDEOS_PER_ROW):
            with gr.Row():
                # Create video components in each row
                for j in range(VIDEOS_PER_ROW):
                    # Make each video player invisible initially
                    video = gr.Video(visible=False, label=f"Video_{i + j}", interactive=False)
                    video_outputs.append(video)

    # Connect the inputs to the function and the output grid
    # The 'change' event triggers the function whenever a filter is changed.
    # The 'outputs' is now a list of all the gr.Video components we created.
    dataset_filter.change(fn=get_videos_for_grid, inputs=[dataset_filter, status_filter], outputs=video_outputs)
    status_filter.change(fn=get_videos_for_grid, inputs=[dataset_filter, status_filter], outputs=video_outputs)

    # Trigger the function once on initial load to populate the gallery
    demo.load(fn=get_videos_for_grid, inputs=[dataset_filter, status_filter], outputs=video_outputs)


demo.launch(share=True)
