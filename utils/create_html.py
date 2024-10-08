import os

def generate_html_for_images(base_dir, output_html):
    # HTML header
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Generated Images</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .folder { margin-bottom: 40px; }
            h2 { color: #333; }
            table { width: 100%; border-collapse: collapse; }
            td { padding: 10px; text-align: center; vertical-align: top; }
            img { max-width: 150px; height: auto; display: block; margin: 0 auto; }
            .image-name { word-wrap: break-word; max-width: 150px; margin-top: 5px; }
        </style>
    </head>
    <body>
    <h1>Generated Images</h1>
    """

    # Traverse the base directory and its subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Sort directories and files
        dirs.sort()
        files.sort()
        
        # Filter image files (only '.png' images)
        images = [f for f in files if f.endswith('.png')]
        if images:
            relative_path = os.path.relpath(root, base_dir)
            folder_name = os.path.basename(root)
            
            html_content += f"<div class='folder'>\n<h2>Folder: {relative_path}</h2>\n<table>\n<tr>\n"
            
            for i, image_name in enumerate(images):
                image_path = os.path.join(relative_path, image_name)
                html_content += f"""
                <td>
                    <img src='{image_path}' alt='{image_name}'>
                    <div class='image-name'>{image_name}</div>
                </td>
                """
                # Break the row after every 5 images
                if (i + 1) % 5 == 0:
                    html_content += "</tr>\n<tr>\n"
            html_content += "</tr>\n</table>\n</div>\n"

    # HTML footer
    html_content += """
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(output_html, 'w') as file:
        file.write(html_content)

# Specify the directory containing the generated images and the output HTML file
base_directory = '.'
output_html_file = 'generated_images.html'

# Generate the HTML file
generate_html_for_images(base_directory, output_html_file)

print(f"HTML file '{output_html_file}' has been generated.")
