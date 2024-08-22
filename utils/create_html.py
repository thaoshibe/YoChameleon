import json

# Load the JSON data
json_file = "/mnt/localssd/code/data/minibo/inpainted.json"
with open(json_file, 'r') as file:
    data = json.load(file)

# Start the HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Conversations</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .image-conversation {
            margin-bottom: 40px;
        }
        .image-conversation img {
            max-width: 100%;
            height: auto;
        }
        .conversation {
            margin-top: 10px;
        }
        .conversation span {
            font-weight: bold;
        }
        .conversation .human {
            color: blue;
        }
        .conversation .bot {
            color: green;
        }
    </style>
</head>
<body>
"""

# Add the image and conversation data
for item in data:
    image_path = item["image"][0].replace('/mnt/localssd/code/data/minibo/', './')
    conversations = item["conversations"]

    html_content += f'<div class="image-conversation">'
    html_content += f'<img src="{image_path}" alt="Image" width="300"/>'
    
    for conversation in conversations:
        speaker = conversation["from"]
        text = conversation["value"].replace("<sks>", "sks")
        
        speaker_class = "human" if speaker == "human" else "bot"
        html_content += f'<div class="conversation">'
        html_content += f'<span class="{speaker_class}">{speaker.capitalize()}:</span> {text}'
        html_content += f'</div>'
    
    html_content += f'</div>'

# Close the HTML content
html_content += """
</body>
</html>
"""

# Save the HTML content to a file
output_html_file = "/mnt/localssd/code/data/minibo/bo-inpainting.html"
with open(output_html_file, 'w') as file:
    file.write(html_content)

print(f"HTML file created: {output_html_file}")
