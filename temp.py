import imageio
import os

def create_video_imageio(image_folder, output_file, fps=25):
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith(".jpg")
    ])

    if not image_files:
        print("❌ No images found in folder.")
        return

    writer = imageio.get_writer(output_file, fps=fps)

    for file in image_files:
        try:
            img = imageio.imread(file)
            writer.append_data(img)
        except Exception as e:
            print(f"⚠️ Failed to write {file}: {e}")
    writer.close()
    print(f"✅ Video saved as {output_file}")

# Example usage
if __name__ == "__main__":
    create_video_imageio("alert_images", "output_annotated2.mp4", fps=25)
