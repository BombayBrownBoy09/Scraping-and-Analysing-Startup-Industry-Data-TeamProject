mkdir -p ~/.streamlit/
echo "[general]
email = \"omartinez1821992@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml