import json
import pandas as pd


# Function to process the JSON data and convert it into a DataFrame
def process_json_to_csv(json_filepath, csv_filepath):
    with open(json_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize lists to hold data rows for issues and comments
    issues_data = []
    comments_data = []

    for issue_id, issue_details in data.items():
        # Extract issue details
        issues_data.append({
            'Issue_ID': issue_id,
            'Title': issue_details.get('title', ''),
            'Body': issue_details.get('body', ''),
            'Created_At': issue_details.get('created_at', ''),
            'Closed_At': issue_details.get('closed_at', ''),
            'Num_Comments': issue_details.get('num_comments', 0),
            'User_ID': issue_details.get('userid', ''),
            'User_Login': issue_details.get('userlogin', '')
        })

        # Extract comments for the issue
        for comment_id, comment_details in issue_details.get('comments', {}).items():
            comments_data.append({
                'Issue_ID': issue_id,
                'Comment_ID': comment_id,
                'Comment_Body': comment_details.get('body', ''),
                'Comment_User_ID': comment_details.get('userid', ''),
                'Comment_User_Login': comment_details.get('userlogin', '')
            })

    # Convert lists to DataFrames
    issues_df = pd.DataFrame(issues_data)
    comments_df = pd.DataFrame(comments_data)

    # Optionally, you can merge these DataFrames or keep them separate depending on your analysis needs
    # For this example, we'll keep them separate and save to CSV files
    issues_df.to_csv(csv_filepath.format('issues'), index=False)
    comments_df.to_csv(csv_filepath.format('comments'), index=False)


# Replace 'path_to_your_large_json_file.json' with the path to your JSON file
# The script expects a format string for csv_filepath where {} will be replaced with 'issues' or 'comments'
process_json_to_csv('powertoys.json', 'powertoys.csv')
