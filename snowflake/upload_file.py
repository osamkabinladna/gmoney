import snowflake.connector

# Define your Snowflake connection parameters
connection_parameters = {
    "user": "obamabinbiden",
    "password": "!Jokow0w",
    "account": "tg20110.ap-southeast-3.aws",
    "warehouse": "COMPUTE_WH",
    "database": "streamlit_demo",
    "schema": "streamlit_schema",
}

# Establish a connection to Snowflake
conn = snowflake.connector.connect(**connection_parameters)

# Create a cursor object
cursor = conn.cursor()

# Specify the local file path and the stage name
local_file_path = "/Users/gregruyoga/gmoneycodes/tradingstrats/random_forest/models/random_forest_data.pkl"
stage_name = "streamlit_stage"

# Upload the file to the Snowflake stage
try:
    cursor.execute(f"PUT file://{local_file_path} @{stage_name} AUTO_COMPRESS=TRUE")
    print(f"File {local_file_path} successfully uploaded to stage {stage_name}.")
except Exception as e:
    print(f"Failed to upload file: {e}")

# Close the cursor and connection
cursor.close()
conn.close()