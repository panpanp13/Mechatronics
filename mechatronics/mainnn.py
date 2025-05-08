import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import json
import os
from detection import run

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
src_pts=[]
# -----------------------------
# Configuration and Constants
# -----------------------------
TARGET_WIDTH  = 25    
TARGET_HEIGHT = 250
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1]
], dtype=np.float32)

PRESET_FOLDER = "preset"
if not os.path.exists(PRESET_FOLDER):
    os.makedirs(PRESET_FOLDER)

#VIDEO_PATH = "test_video/vehicles.mp4"
VIDEO_PATH = 1 # define file path to receive camera frame 

# -----------------------------
# Global Variables
# -----------------------------
lane_lines = []        # Each: {"x_target": float, "pt1": (x,y), "pt2": (x,y)}
display_frame = None   # Resized first frame (BGR)
SOURCE_scaled = None   # Scaled ROI polygon (for 1280Ã—720)
global_photo = None    # Tkinter PhotoImage reference
canvas_global = None   # Tkinter canvas reference
inv_transform = None   # Inverse perspective transform (from warped to original)
M = None               # Forward perspective transform (from original to warped)
traffic_var = None     # IntVar for Traffic Island checkbox
island_polygon = None  # Final island polygon (list of (x,y) tuples) in original image
pattern_image = None   # Final pattern in original perspective (1280Ã—720)

drag_locked = False    # When True, dragging is disabled
selected_line_idx = None
last_mouse_x = None

preset_loaded = False      # When a preset is loaded, no modifications are allowed
traffic_island_mode = False  # Stores island mode state as of Generate time

action_button = None   # This button will change between "Lock", "Next", and "Start"
instr_label = None
gui_root = None
pick_win=None
preset_var = None      # Tkinter StringVar for preset selection
option_menu = None

lane_entry = None 
# -----------------------------
# Perspective Helper Functions
# -----------------------------
def transform_point(pt, matrix):
    src = np.array([[pt]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)
    return float(dst[0,0,0]), float(dst[0,0,1])

def warped_line_to_original(x_target):
    top_warped = (x_target, 0)
    bot_warped = (x_target, TARGET_HEIGHT - 1)
    pt1 = transform_point(top_warped, inv_transform)
    pt2 = transform_point(bot_warped, inv_transform)
    return (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]))

def get_line_midpoint_in_original_space(x_target):
    pt1, pt2 = warped_line_to_original(x_target)
    return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

# -----------------------------
# Lane Generation and Overlay
# -----------------------------
def generate_lane_lines(num_lanes: int, traffic_island: bool):
    global lane_lines, island_polygon
    lane_lines = []
    island_polygon = None
    total_boundaries = num_lanes + (2 if traffic_island else 1)
    for i in range(total_boundaries):
        x_t = i * ((TARGET_WIDTH - 1) / (total_boundaries - 1))
        pt1, pt2 = warped_line_to_original(x_t)
        lane_lines.append({"x_target": x_t, "pt1": pt1, "pt2": pt2})

def overlay_pattern(image, pattern, polygon):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    section = cv2.bitwise_and(pattern, pattern, mask=mask)
    return cv2.addWeighted(image, 1.0, section, 1.0, 0)

# -----------------------------
# Canvas Update & Events
# -----------------------------
def update_canvas(canvas, base_image):
    global canvas_global
    temp = base_image.copy()
    if island_polygon and pattern_image is not None:
        temp = overlay_pattern(temp, pattern_image, island_polygon)
    color = (255,255,255) if drag_locked else (0,181,247)
    for line in lane_lines:
        cv2.line(temp, line['pt1'], line['pt2'], color, 3)
    rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(rgb)
    canvas_global.image = ImageTk.PhotoImage(im_pil)
    canvas_global.itemconfig('img', image=canvas_global.image)
# -----------------------------
# Mouse Drag for Boundary Adjustment
# -----------------------------
def on_canvas_drag_start(event):
    global selected_line_idx, last_mouse_x
    # If preset is loaded or dragging is locked, do not allow modifications.
    if preset_loaded or drag_locked:
        return

    candidate = None
    min_dist = 999999
    for i, line in enumerate(lane_lines):
        # REMOVE the check that prevented moving the first/last boundaries
        d = point_line_distance(event.x, event.y,
                                line["pt1"][0], line["pt1"][1],
                                line["pt2"][0], line["pt2"][1])
        if d < 10 and d < min_dist:
            min_dist = d
            candidate = i

    if candidate is not None:
        selected_line_idx = candidate
        last_mouse_x = event.x

def on_canvas_drag_motion(event):
    global selected_line_idx, last_mouse_x
    if preset_loaded or drag_locked:
        return
    if selected_line_idx is not None and last_mouse_x is not None:
        dx = event.x - last_mouse_x
        if dx == 0:
            return

        old_mid = get_line_midpoint_in_original_space(lane_lines[selected_line_idx]["x_target"])
        new_mid_orig = (old_mid[0] + dx, old_mid[1])
        old_mid_warped = transform_point(old_mid, M)
        new_mid_warped = transform_point(new_mid_orig, M)
        dx_warped = new_mid_warped[0] - old_mid_warped[0]
        lane_lines[selected_line_idx]["x_target"] += dx_warped
        # Keep x_target within [0, TARGET_WIDTH - 1]
        lane_lines[selected_line_idx]["x_target"] = max(0,
            min(lane_lines[selected_line_idx]["x_target"], TARGET_WIDTH - 1))
        pt1, pt2 = warped_line_to_original(lane_lines[selected_line_idx]["x_target"])
        lane_lines[selected_line_idx]["pt1"] = pt1
        lane_lines[selected_line_idx]["pt2"] = pt2
        last_mouse_x = event.x
        update_canvas(canvas_global, display_frame)

def on_canvas_drag_release(event):
    global selected_line_idx, last_mouse_x
    if preset_loaded:
        return
    selected_line_idx = None
    last_mouse_x = None
    
def point_line_distance(px, py, x1, y1, x2, y2):
    lv = np.array([x2-x1, y2-y1])
    pv = np.array([px-x1, py-y1])
    ll2 = lv.dot(lv)
    if ll2 < 1e-12: return float(np.linalg.norm(pv))
    t = max(0, min(1, pv.dot(lv) / ll2))
    proj = np.array([x1,y1]) + t*lv
    return float(np.linalg.norm(pv - t*lv))

def on_canvas_right_click(event):
    if preset_loaded or (not drag_locked or not traffic_island_mode):
        return

    click_warped = transform_point((event.x, event.y), M)
    click_x = click_warped[0]
    x_targets = [line["x_target"] for line in lane_lines]
    if click_x < x_targets[0] or click_x > x_targets[-1]:
        return
    left = None
    right = None
    for x in x_targets:
        if x <= click_x:
            left = x
        if x >= click_x:
            right = x
            break
    if left is None or right is None or left == right:
        return
    # Construct ideal island rectangle in warped space.
    rect_warped = np.array([
        [left, 0],
        [right, 0],
        [right, TARGET_HEIGHT - 1],
        [left, TARGET_HEIGHT - 1]
    ], dtype=np.float32)
    island_ideal = []
    for pt in rect_warped:
        p = transform_point(tuple(pt), inv_transform)
        island_ideal.append([p[0], p[1]])
    island_ideal = np.array(island_ideal, dtype=np.float32)

    # Optionally, intersect with ROI to ensure island doesn't go outside.
    roi_poly = SOURCE_scaled.reshape((-1, 2)).astype(np.float32)
    retval, intersect_poly = cv2.intersectConvexConvex(roi_poly, island_ideal)
    if retval > 0 and intersect_poly is not None:
        global island_polygon
        island_polygon = [(int(p[0]), int(p[1])) for p in intersect_poly.reshape(-1, 2)]
        update_canvas(canvas_global, display_frame)

        # Check which lane the island region is between
        island_x_min = min([p[0] for p in island_polygon])
        island_x_max = max([p[0] for p in island_polygon])
        for idx, x_target in enumerate(x_targets[:-1]):
            if x_targets[idx] <= island_x_min and x_targets[idx + 1] >= island_x_max:
                print(f"Island is between lane {idx + 1} and lane {idx + 2}")
                break

        instr_label.config(text="Island region marked. Click Start to finish.")
        action_button.grid(row=1, column=4, padx=75, pady=5)
        action_button.config(text="Start", state="normal", command=start_detection)

# -----------------------------
# Next/Start Button and Lock Callback
# -----------------------------
def next_or_start():
    global drag_locked
    if not traffic_island_mode:
        gui_root.destroy()
    else:
        if not drag_locked:
            drag_locked = True
            update_canvas(canvas_global, display_frame)
            instr_label.config(text="Dragging locked. Now, right-click in the inner region to mark the island.")
            action_button.grid(row=1, column=4, padx=75, pady=5)
            action_button.config(text="Next", state="disabled")
        else:
            gui_root.destroy()

def lock_dragging():
    global drag_locked
    drag_locked = True
    update_canvas(canvas_global, display_frame)
    instr_label.config(text="Dragging locked. Click Start to finish.")
    action_button.config(text="Start", command=start_detection)

# -----------------------------
# on_generate Callback with Lane Number Validation
# -----------------------------
def on_generate(lanes_entry, traffic_var_, canvas, base_image):
    global action_button, island_polygon, drag_locked, pattern_image, preset_loaded, traffic_island_mode
    # When generating, reset preset flag.
    preset_loaded = False

    # Validate lane number input: must be a positive integer.
    try:
        num_lanes = int(lanes_entry.get())
        if num_lanes <= 0:
            messagebox.showerror("Invalid Input", "Lane number must be a positive integer.")
            return
    except ValueError:
        messagebox.showerror("Invalid Input", "Lane number must be a positive integer.")
        return

    # Store island mode state from checkbox at generate time.
    traffic_island_mode = bool(traffic_var_.get())
    generate_lane_lines(num_lanes, traffic_island_mode)
    island_polygon = None
    drag_locked = False
    update_canvas(canvas, base_image)
    # Create the pattern on a 1000x1000 square and then warp it into ROI.
    h, w = base_image.shape[:2]
    pattern_image = create_square_pattern_in_roi((w, h))
    action_button.grid(row=1, column=4, padx=75, pady=5)
    if traffic_island_mode:
        action_button.config(text="Next", state="normal", command=next_or_start)
        instr_label.config(text="Adjust lane boundaries as needed. Then click Next to lock dragging and mark the island.")
    else:
        action_button.config(text="Lock", state="normal", command=lock_dragging)
        instr_label.config(text="Adjust lane boundaries as needed. Then click Lock to lock dragging.")

    # Set the lane entry to readonly (no blinking cursor)
    lanes_entry.config(state="readonly")
    # Bind a click event to re-enable editing when user clicks the entry box again.
    lanes_entry.bind("<Button-1>", enable_lane_entry)

def enable_lane_entry(event):
    """Re-enable the lane entry when the user clicks on it."""
    event.widget.config(state="normal")
    # Unbind this event so that further clicks don't re-enable repeatedly.
    event.widget.unbind("<Button-1>")

# -----------------------------
# Create Square Pattern in ROI
# -----------------------------
def create_square_pattern():
    size = 1000
    pattern = np.zeros((size, size, 3), dtype=np.uint8)
    spacing = 15  # adjust spacing as needed
    thickness = 2
    color = (0, 90, 150)
    for i in range(-size, size, spacing):
        pt1 = (i, 0)
        pt2 = (i + size, size)
        cv2.line(pattern, pt1, pt2, color, thickness)
    return pattern

def create_square_pattern_in_roi(out_size):
    square_pattern = create_square_pattern()
    square = np.array([[0, 0], [999, 0], [999, 999], [0, 999]], dtype=np.float32)
    roi = SOURCE_scaled.reshape((-1, 2)).astype(np.float32)
    T = cv2.getPerspectiveTransform(square, roi)
    pattern_in_roi = cv2.warpPerspective(square_pattern, T, out_size, flags=cv2.INTER_LANCZOS4)
    return pattern_in_roi

# -----------------------------
# Preset Save/Load Functions
# -----------------------------
def save_preset():
    preset = {
        "lane_lines": lane_lines,
        "island_polygon": island_polygon
    }
    file_path = filedialog.asksaveasfilename(
        initialdir=PRESET_FOLDER,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        title="Save Preset"
    )
    if file_path:
        with open(file_path, "w") as f:
            json.dump(preset, f)
        messagebox.showinfo("Save Preset", "Preset saved successfully!")
        refresh_preset_list()

def load_preset_from_file(preset_filename):
    global lane_lines, island_polygon, drag_locked, pattern_image, preset_loaded
    full_path = os.path.join(PRESET_FOLDER, preset_filename)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            preset = json.load(f)
        lane_lines = preset.get("lane_lines", [])
        island_polygon = preset.get("island_polygon", None)
        for line in lane_lines:
            line["pt1"] = tuple(line["pt1"])
            line["pt2"] = tuple(line["pt2"])
        h, w = display_frame.shape[:2]
        pattern_image = create_square_pattern_in_roi((w, h))
        drag_locked = True
        preset_loaded = True
        update_canvas(canvas_global, display_frame)
        action_button.grid(row=1, column=4, padx=75, pady=5)
        action_button.config(text="Start", state="normal", command=start_detection)  # Ensure 'Start' runs detection
        instr_label.config(text="Preset loaded. No modifications allowed. Click Start to finish.")
        messagebox.showinfo("Load Preset", "Preset loaded successfully!")

def refresh_preset_list():
    global preset_var, option_menu
    presets = [f for f in os.listdir(PRESET_FOLDER) if f.endswith(".json")]
    if not presets:
        presets = ["<None>"]
    preset_var.set(presets[0])
    option_menu["menu"].delete(0, "end")
    for p in presets:
        option_menu["menu"].add_command(label=p, command=tk._setit(preset_var, p))

def load_preset_from_menu():
    preset_filename = preset_var.get()
    if preset_filename and preset_filename != "<None>":
        load_preset_from_file(preset_filename)
    else:
        messagebox.showwarning("Load Preset", "No preset file selected.")
    
def start_detection():
    global island_polygon, lane_lines, VIDEO_PATH,src_pts
    try:
        if island_polygon is not None:
            # Compute the centroid of the island polygon in original coordinates.
            island_np = np.array(island_polygon, dtype=np.float32)
            centroid = np.mean(island_np, axis=0)
            # Transform the centroid to warped space.
            centroid_warped = transform_point((centroid[0], centroid[1]), M)
            cx = centroid_warped[0]
            lane_number = None
            # Loop over lane boundaries to find the interval containing the centroid.
            # (Assuming lane boundaries are numbered starting at 1.)
            for i in range(len(lane_lines) - 1):
                left_boundary = lane_lines[i]["x_target"]
                right_boundary = lane_lines[i+1]["x_target"]
                if left_boundary <= cx <= right_boundary:
                    lane_number = i + 1  # Convert to 1-indexed lane number.
                    break
            run(lane_lines, lane_number, VIDEO_PATH,src_pts)
        else:
            run(lane_lines, 0, VIDEO_PATH,src_pts)
    except Exception as e:
        print("Error in detection:", e)

# -----------------------------
# Main Function
# -----------------------------
def main():
    global display_frame, SOURCE_scaled, canvas_global, inv_transform, M, traffic_var
    global action_button, instr_label, gui_root, preset_var, option_menu, drag_locked, preset_loaded, lane_entry,src_pts
    # 1) Read & resize first frame

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error reading video")
        return
    tw, th = 640,360
    display_frame = cv2.resize(frame, (tw, th))

    # 2) User picks 4 ROI points
    src_pts = []
    pick_win = tk.Tk()
    pick_win.title("Click 4 ROI corners")
    rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(rgb))
    cv = tk.Canvas(pick_win, width=tw, height=th)
    cv.pack()
    cv.create_image(0,0,anchor='nw',image=img)
    def click(evt):
        src_pts.append((evt.x, evt.y))
        r=5
        cv.create_oval(evt.x-r, evt.y-r, evt.x+r, evt.y+r, fill='red')
        if len(src_pts)==4:
            cv.unbind('<Button-1>')
            pick_win.destroy()
    cv.bind('<Button-1>', click)
    pick_win.mainloop()
    if len(src_pts)<4:
        print("ROI selection cancelled")
        return
    print("Selected ROI points (clockwise):", src_pts)
    messagebox.showinfo("ROI Selected",
        "You clicked:\n" +
        "\n".join(f"({x}, {y})" for x,y in src_pts)
    )

    SOURCE_scaled = np.array(src_pts, dtype=np.float32)

    # 3) Compute transforms
    M = cv2.getPerspectiveTransform(SOURCE_scaled, TARGET)
    inv_transform = cv2.getPerspectiveTransform(TARGET, SOURCE_scaled)

    # Optional debug overlay
    frame_controls = tk.Frame(gui_root)
    frame_controls.grid(row=0, column=0, columnspan=5, pady=10)

    # Canvas (display image) setup below the control bar
    canvas = tk.Canvas(gui_root, width=tw, height=th)
    canvas.grid(row=1, column=0, columnspan=5)

    canvas_global = canvas

    temp_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(temp_rgb)
    global global_photo
    global_photo = ImageTk.PhotoImage(image=im_pil)
    canvas.create_image(0, 0, anchor="nw", image=global_photo, tags="img")

    canvas.bind("<ButtonPress-1>", on_canvas_drag_start)
    canvas.bind("<B1-Motion>", on_canvas_drag_motion)
    canvas.bind("<ButtonRelease-1>", on_canvas_drag_release)
    canvas.bind("<Button-3>", on_canvas_right_click)

    instr_label = tk.Label(frame_controls, text="Adjust lane boundaries as needed.")
    instr_label.grid(row=0, column=0, columnspan=5, pady=5)

    tk.Label(frame_controls, text="Number of lanes:").grid(row=1, column=0, padx=5, pady=5)
    lane_entry = tk.Entry(frame_controls)
    lane_entry.grid(row=1, column=1, padx=5, pady=5)
    lane_entry.insert(0, "")

    # Traffic Island Checkbox
    global traffic_var
    traffic_var = tk.IntVar()
    tk.Checkbutton(frame_controls, text="Traffic Island", variable=traffic_var).grid(row=1, column=2, padx=75, pady=5)

    gen_btn = tk.Button(frame_controls, text="Generate",
                        command=lambda: on_generate(lane_entry, traffic_var, canvas, display_frame))
    gen_btn.grid(row=1, column=3, padx=75, pady=5)

    # Action Button (Lock/Next/Start) will be repositioned depending on mode.
    action_button = tk.Button(frame_controls, text="Next", state="disabled", command=start_detection)
    action_button.grid(row=1, column=4, padx=75, pady=5)

    # Preset OptionMenu and Load Button
    preset_var = tk.StringVar()
    preset_var.set("<None>")
    tk.Label(frame_controls, text="Preset:").grid(row=2, column=0, padx=5, pady=5)
    option_menu = tk.OptionMenu(frame_controls, preset_var, "<None>")
    option_menu.grid(row=2, column=1, padx=75, pady=5)
    refresh_preset_list()
    load_preset_btn = tk.Button(frame_controls, text="Load Preset", command=load_preset_from_menu)
    load_preset_btn.grid(row=2, column=2, padx=75, pady=5)

    # Save Preset Button (saves via a file dialog into the preset folder)
    save_button = tk.Button(frame_controls, text="Save Preset", command=save_preset)
    save_button.grid(row=2, column=3, padx=75, pady=5)

    pick_win.mainloop()

if __name__ == "__main__":
    main()