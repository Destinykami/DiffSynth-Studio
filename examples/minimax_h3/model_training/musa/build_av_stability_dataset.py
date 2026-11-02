"""Build a 16-sample mixed (real + synthetic multi-pattern) audio-visual benchmark dataset
for MiniMax-H3 long-run stability and loss-convergence verification.

Layout produced under `OUT_DIR`:
  metadata.csv   (columns: video,prompt,input_audio,frame_rate)
  sample_00_real.mp4 (real video from MiniMax-H3-FL2VA example dataset)
  sample_01_*.mp4 .. sample_15_*.mp4 (15 distinct audiovisual clips: 832x480, 24fps, 124 frames,
                                      32kHz stereo AAC audio, variable-length prompts)
"""
import csv
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
OUT_DIR = Path(os.environ.get("AV_BENCH_DIR", str(REPO_ROOT / "data/av_stability_dataset")))
REAL_MP4 = Path(
    os.environ.get(
        "REAL_MP4_PATH",
        str(REPO_ROOT / "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/video.mp4"),
    )
)
REAL_PROMPT = (
    'A girl is very happy, she is speaking in english: '
    '"I enjoy working with Diffsynth-Studio, it\'s a perfect framework."'
)

SYNTH_SPECS = [
    (
        "red_right_220hz",
        "0x101820",
        "0xff3b30",
        "40+t*110",
        "180",
        220,
        440,
        "A bright red square moves smoothly from left to right across a dark slate background while a warm 220 Hz and 440 Hz stereo harmonic tone plays.",
    ),
    (
        "green_left_330hz",
        "0x182018",
        "0x34c759",
        "680-t*110",
        "180",
        330,
        660,
        "A vivid green block glides from right to left on a deep forest canvas accompanied by a clear 330 Hz stereo musical chord.",
    ),
    (
        "blue_down_440hz",
        "0x101428",
        "0x007aff",
        "356",
        "30+t*55",
        440,
        880,
        "A cobalt blue rectangle descends vertically down the center of the frame synchronized with a pure 440 Hz concert pitch stereo sine wave.",
    ),
    (
        "gold_up_528hz",
        "0x241c10",
        "0xffcc00",
        "356",
        "330-t*55",
        528,
        264,
        "A golden yellow box rises steadily from the bottom to the top of a warm brown stage with a 528 Hz and 264 Hz stereo duet.",
    ),
    (
        "cyan_diag_dr_392hz",
        "0x0c2226",
        "0x5ac8fa",
        "60+t*100",
        "40+t*50",
        392,
        588,
        "A cyan square travels diagonally from the top-left corner toward the bottom-right corner while a 392 Hz G-major fifth interval resonates in stereo.",
    ),
    (
        "magenta_diag_ul_494hz",
        "0x260c22",
        "0xff2d55",
        "660-t*100",
        "320-t*50",
        494,
        741,
        "A magenta marker moves diagonally upward from bottom-right to top-left across a dark plum backdrop with a 494 Hz and 741 Hz stereo chime.",
    ),
    (
        "orange_orbit_294hz",
        "0x22160c",
        "0xff9500",
        "356+220*cos(2*t)",
        "180+110*sin(2*t)",
        294,
        588,
        "An orange square orbits in a smooth circular trajectory around the center of the screen while a 294 Hz D-note octave plays continuously in both audio channels.",
    ),
    (
        "purple_osc_x_349hz",
        "0x1a1028",
        "0xaf52de",
        "356+260*sin(3*t)",
        "180",
        349,
        523,
        "A purple square oscillates horizontally back and forth across the middle of a violet background with a 349 Hz and 523 Hz F-major stereo interval.",
    ),
    (
        "teal_osc_y_262hz",
        "0x0e2420",
        "0x30b0c7",
        "356",
        "180+130*cos(3*t)",
        262,
        392,
        "A teal block bounces vertically in a cosine wave motion at the center of the video frame while a 262 Hz middle-C and 392 Hz G fifth sound in stereo.",
    ),
    (
        "lime_fast_right_659hz",
        "0x14200e",
        "0xa4ff00",
        "mod(t*220,700)",
        "100",
        659,
        330,
        "A neon lime square sweeps rapidly from left to right in repeating cycles near the top of the frame with a bright 659 Hz E-note stereo tone.",
    ),
    (
        "coral_fast_left_587hz",
        "0x241214",
        "0xff6b6b",
        "700-mod(t*220,700)",
        "260",
        587,
        294,
        "A coral pink block sweeps rapidly from right to left along the lower third of the frame accompanied by a 587 Hz and 294 Hz stereo harmonic.",
    ),
    (
        "white_center_pulse_440_660",
        "0x121212",
        "0xf2f2f7",
        "356+120*sin(4*t)",
        "180+80*cos(4*t)",
        440,
        660,
        "A crisp white square traces a fast elliptical loop around the center of a charcoal background while a balanced 440 Hz and 660 Hz stereo fifth plays.",
    ),
    (
        "amber_fig8_370hz",
        "0x1f1a0e",
        "0xffb340",
        "356+240*sin(2*t)",
        "180+110*sin(4*t)",
        370,
        555,
        "An amber square traces a figure-eight Lissajous curve across the frame while a steady 370 Hz and 555 Hz stereo chord rings out.",
    ),
    (
        "sky_slow_drift_247hz",
        "0x0c1828",
        "0x64d2ff",
        "120+t*80",
        "280-t*35",
        247,
        494,
        "A sky-blue square drifts gently upward and to the right across a deep navy background paired with a low 247 Hz and 494 Hz stereo octave.",
    ),
    (
        "rose_spiral_415hz",
        "0x240e1a",
        "0xff375f",
        "356+180*cos(2.5*t)",
        "180-100*sin(2.5*t)",
        415,
        622,
        "A rose-red square rotates counter-clockwise around the center of a burgundy canvas while a 415 Hz and 622 Hz stereo harmony plays smoothly.",
    ),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    if REAL_MP4.exists():
        s0_name = "sample_00_real.mp4"
        shutil.copy2(REAL_MP4, OUT_DIR / s0_name)
        rows.append((s0_name, REAL_PROMPT, s0_name, 24))

    duration = 124.0 / 24.0
    for idx, (tag, bg_hex, fg_hex, x_expr, y_expr, f_left, f_right, prompt) in enumerate(SYNTH_SPECS, start=1):
        fname = f"sample_{idx:02d}_{tag}.mp4"
        fpath = OUT_DIR / fname
        if not fpath.exists():
            vf = (
                f"color=c={bg_hex}:s=832x480:r=24,"
                f"drawbox=x='{x_expr}':y='{y_expr}':w=120:h=120:color={fg_hex}@1.0:t=fill"
            )
            af = f"aevalsrc='0.25*sin(2*PI*{f_left}*t)|0.25*sin(2*PI*{f_right}*t)':s=32000:d={duration:.6f}"
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                vf,
                "-f",
                "lavfi",
                "-i",
                af,
                "-frames:v",
                "124",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "32000",
                "-ac",
                "2",
                str(fpath),
            ]
            subprocess.run(cmd, check=True)
        rows.append((fname, prompt, fname, 24))

    meta_path = OUT_DIR / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "prompt", "input_audio", "frame_rate"])
        writer.writerows(rows)

    print(f"[build_av_stability_dataset] Created {len(rows)} samples in {OUT_DIR} (metadata: {meta_path})")


if __name__ == "__main__":
    main()
