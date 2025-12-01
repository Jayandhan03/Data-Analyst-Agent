from ChatAgent import Chat_Data_Agent

# Example usage
response = Chat_Data_Agent(df_path="Bakery sales.csv", query="Which product sold the most?")
print(response['response'])