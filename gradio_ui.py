#!/usr/bin/env python3
"""
SeedVR2-3B Web UI using Gradio
macOS compatible version with MPS support and CPU fallback
"""

import gradio as gr
import os
import sys
import tempfile
import shutil
from pathlib import Path
import datetime
import traceback

# Add SeedVR to path
sys.path.insert(0, '/Users/shikigami/opensource/SeedVR')

# Import our working inference script functions
try:
    from projects.inference_seedvr2_3b_macos import configure_runner, generation_loop, is_image_file
    print("✅ Successfully imported SeedVR inference functions")
except Exception as e:
    print(f"❌ Error importing SeedVR functions: {e}")
    sys.exit(1)

# Global runner instance
runner = None

def initialize_model():
    """Initialize the SeedVR model - this takes time so we do it once"""
    global runner
    if runner is None:
        print("🔧 Initializing SeedVR2-3B model...")
        try:
            runner = configure_runner(sp_size=1)
            print("✅ Model initialized successfully!")
            return "✅ Model loaded successfully!"
        except Exception as e:
            error_msg = f"❌ Error initializing model: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg
    else:
        return "✅ Model already loaded!"

def enhance_media(
    input_file,
    output_resolution,
    cfg_scale,
    cfg_rescale, 
    sample_steps,
    seed,
    out_fps,
    progress=gr.Progress()
):
    """Main function to enhance images or videos"""
    global runner
    
    if runner is None:
        return None, "❌ Model not initialized. Please click 'Initialize Model' first."
    
    if input_file is None:
        return None, "❌ Please upload an image or video file."
    
    try:
        # Create temporary directories
        temp_input_dir = tempfile.mkdtemp(prefix="seedvr_input_")
        temp_output_dir = tempfile.mkdtemp(prefix="seedvr_output_")
        
        progress(0.1, "Setting up files...")
        
        # Copy input file to temp directory
        input_path = Path(input_file)
        temp_input_path = Path(temp_input_dir) / input_path.name
        shutil.copy2(input_file, temp_input_path)
        
        # Determine resolution
        if output_resolution == "Auto (from input)":
            res_h, res_w = None, None
        else:
            res_w, res_h = map(int, output_resolution.split('x'))
        
        # Set fps
        fps_val = None if out_fps == "Auto (from input)" else float(out_fps)
        
        progress(0.2, "Starting enhancement...")
        
        # Run the enhancement
        generation_loop(
            runner=runner,
            video_path=temp_input_dir,
            output_dir=temp_output_dir,
            batch_size=1,
            cfg_scale=cfg_scale,
            cfg_rescale=cfg_rescale,
            sample_steps=sample_steps,
            seed=seed,
            res_h=res_h,
            res_w=res_w,
            sp_size=1,
            out_fps=fps_val
        )
        
        progress(0.9, "Finalizing output...")
        
        # Find the output file
        output_files = list(Path(temp_output_dir).glob("*"))
        if not output_files:
            return None, "❌ No output file generated."
        
        output_file = output_files[0]
        
        # Copy to a permanent location for Gradio
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_dir = Path("./outputs")
        final_output_dir.mkdir(exist_ok=True)
        
        file_ext = output_file.suffix
        final_output_path = final_output_dir / f"enhanced_{timestamp}{file_ext}"
        shutil.copy2(output_file, final_output_path)
        
        # Cleanup temp directories
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        
        progress(1.0, "Complete!")
        
        return str(final_output_path), f"✅ Enhancement completed successfully!\nOutput saved to: {final_output_path}"
        
    except Exception as e:
        error_msg = f"❌ Error during enhancement: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        # Cleanup on error
        try:
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        except:
            pass
            
        return None, error_msg

# Create the Gradio interface
def create_ui():
    with gr.Blocks(
        title="SeedVR2-3B Video/Image Enhancer",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎬 SeedVR2-3B Video & Image Enhancer</h1>
            <p>AI-powered super-resolution for videos and images using a 3.4B parameter diffusion model</p>
            <p><em>macOS optimized with CPU fallback for maximum compatibility</em></p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Input and controls
            with gr.Column(scale=1):
                gr.Markdown("## 📁 Input")
                
                input_file = gr.File(
                    label="Upload Image or Video",
                    file_types=["image", "video"],
                    type="filepath"
                )
                
                # Model initialization
                with gr.Row():
                    init_btn = gr.Button("🚀 Initialize Model", variant="primary", size="lg")
                    init_status = gr.Textbox(
                        label="Model Status",
                        value="❌ Model not loaded",
                        interactive=False
                    )
                
                gr.Markdown("## ⚙️ Enhancement Settings")
                
                with gr.Row():
                    output_resolution = gr.Dropdown(
                        label="Output Resolution",
                        choices=[
                            "Auto (from input)",
                            "720x480",
                            "1280x720", 
                            "1920x1080",
                            "2560x1440",
                            "3840x2160"
                        ],
                        value="Auto (from input)"
                    )
                
                with gr.Row():
                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.1,
                        value=7.5,
                        info="Higher values follow the prompt more closely"
                    )
                    
                    cfg_rescale = gr.Slider(
                        label="CFG Rescale", 
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.0,
                        info="Helps prevent over-saturation"
                    )
                
                with gr.Row():
                    sample_steps = gr.Slider(
                        label="Sampling Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        info="More steps = better quality but slower"
                    )
                    
                    seed = gr.Number(
                        label="Seed",
                        value=666,
                        precision=0,
                        info="For reproducible results"
                    )
                
                with gr.Row():
                    out_fps = gr.Dropdown(
                        label="Output FPS (videos only)",
                        choices=["Auto (from input)", "24", "30", "60"],
                        value="Auto (from input)"
                    )
                
                enhance_btn = gr.Button("✨ Enhance Media", variant="primary", size="lg")
                
            # Right column - Output and preview
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Output")
                
                output_file = gr.File(
                    label="Enhanced Media",
                    interactive=False
                )
                
                output_preview = gr.Image(
                    label="Preview (for images)",
                    interactive=False
                )
                
                output_video = gr.Video(
                    label="Preview (for videos)",
                    interactive=False
                )
                
                output_status = gr.Textbox(
                    label="Status",
                    value="Ready to enhance media",
                    interactive=False,
                    lines=3
                )
        
        # Information section
        with gr.Row():
            gr.Markdown("""
            ## 📋 Instructions
            
            1. **Initialize Model**: Click "🚀 Initialize Model" first (this takes ~10 seconds)
            2. **Upload Media**: Choose an image (.jpg, .png) or video (.mp4, .avi, etc.)
            3. **Adjust Settings**: Configure resolution, quality settings, and sampling steps
            4. **Enhance**: Click "✨ Enhance Media" and wait for processing
            
            ### ⏱️ Processing Times (CPU mode):
            - **Images**: ~30-40 minutes for 50 steps
            - **Short videos**: Varies by length and resolution
            
            ### 💡 Tips:
            - Start with fewer steps (20-30) for faster testing
            - Higher CFG Scale = more dramatic enhancement
            - Use "Auto" resolution to maintain original aspect ratio
            """)
        
        # Event handlers
        def update_preview(file_path, status):
            if file_path is None:
                return None, None, status
            
            try:
                if is_image_file(file_path):
                    return file_path, None, status
                else:
                    return None, file_path, status
            except:
                return None, None, status
        
        def handle_enhancement(input_file, output_resolution, cfg_scale, cfg_rescale, 
                             sample_steps, seed, out_fps, progress=gr.Progress()):
            
            result_file, status = enhance_media(
                input_file, output_resolution, cfg_scale, cfg_rescale,
                sample_steps, seed, out_fps, progress
            )
            
            if result_file:
                # Update preview based on file type
                try:
                    if is_image_file(result_file):
                        return result_file, result_file, None, status
                    else:
                        return result_file, None, result_file, status
                except:
                    return result_file, None, None, status
            else:
                return None, None, None, status
        
        # Wire up the events
        init_btn.click(
            fn=initialize_model,
            outputs=[init_status]
        )
        
        enhance_btn.click(
            fn=handle_enhancement,
            inputs=[
                input_file, output_resolution, cfg_scale, cfg_rescale,
                sample_steps, seed, out_fps
            ],
            outputs=[output_file, output_preview, output_video, output_status]
        )
    
    return demo

if __name__ == "__main__":
    # Create outputs directory
    Path("./outputs").mkdir(exist_ok=True)
    
    print("🌟 Starting SeedVR2-3B Web UI...")
    print("🔧 Note: First-time model initialization will take ~10 seconds")
    print("⏱️  Enhancement typically takes 30-40 minutes on CPU")
    
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )