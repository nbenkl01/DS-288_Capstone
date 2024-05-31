import os
from datetime import datetime
import re

def get_latest_report(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    return files[0] if files else None

def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def update_readme(report_content):
    readme_path = 'README.md'
    current_date = datetime.now().strftime("%Y-%m-%d")
    weekly_status_header = f"## Weekly Status Report ({current_date})\n\n"

    with open(readme_path, 'r') as readme:
        readme_content = readme.read()
    
    updated_content = re.sub(
        r"(## Weekly Status Report.*?\n\n)(.*?)(\n## |$)", 
        f"{weekly_status_header}{report_content}\n\\3", 
        readme_content, 
        flags=re.DOTALL
    )

    # if updated_content == readme_content:
    #     updated_content += weekly_status_header + report_content + "\n"
        
    readme_content = updated_content
    
    with open(readme_path, 'w') as readme:
        readme.write(updated_content)

if __name__ == "__main__":
    latest_report = get_latest_report('StatusReports')
    if latest_report:
        report_content = read_file(os.path.join('StatusReports', latest_report))
        update_readme(report_content)
    else:
        update_readme("No status reports available.")