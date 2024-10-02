mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"tiffanyshi.tang@mail.utoronto.ca\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = true\n\
\n\
" > ~/.streamlit/config.toml