import uvicorn
import webbrowser
import threading
import time

def open_browser():
    """Открывает браузер через небольшую задержку после запуска сервера"""
    time.sleep(2.5)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    # Запускаем открытие браузера в отдельном потоке
    threading.Thread(target=open_browser, daemon=True).start()
    # Запускаем сервер
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)