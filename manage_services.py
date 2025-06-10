# --- manage_services.py ---
# Ein zentrales Skript zum Starten und Stoppen aller Microservices für die RAG-Pipeline.
#
# ANWENDUNG:
#   - Starten aller Dienste in neuen Terminals:
#     python manage_services.py start
#
#   - Stoppen aller laufenden Dienste:
#     python manage_services.py stop
#
# VORAUSSETZUNGEN:
#   - psutil muss installiert sein: pip install psutil
#   - Die Dienstkonfiguration muss in config.json vorhanden sein.

import subprocess
import sys
import os
import json
import argparse
import time
import logging

# --- Logging-Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfiguration laden ---
CONFIG_FILE = "config.json"

def load_config():
    """Lädt die Konfigurationsdatei."""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if "microservices" not in config:
            logging.error(f"Fehler: Der Schlüssel 'microservices' wurde in '{CONFIG_FILE}' nicht gefunden.")
            sys.exit(1)
        return config["microservices"]
    except FileNotFoundError:
        logging.error(f"Fehler: Die Konfigurationsdatei '{CONFIG_FILE}' wurde nicht gefunden.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Fehler: Die Konfigurationsdatei '{CONFIG_FILE}' ist kein gültiges JSON.")
        sys.exit(1)

def start_services(services):
    """Startet alle Dienste in separaten Terminal-Fenstern."""
    logging.info("Starte alle Microservices in neuen Terminal-Fenstern...")
    
    python_executable = sys.executable  # Verwendet das Python-Executable der aktuellen venv
    
    for service in services:
        module = service["module"]
        port = service["port"]
        name = service["name"]
        
        # Baut den uvicorn-Befehl
        command = f'"{python_executable}" -m uvicorn {module}:app --reload --port {port}'
        
        # Betriebssystem-spezifischer Befehl zum Öffnen eines neuen Terminals
        try:
            if sys.platform == "win32":
                # Windows: start cmd /c "command"
                # /K hält das Fenster nach Ausführung offen, /C schließt es.
                # Wir geben dem Fenster einen Titel zur besseren Übersicht.
                subprocess.Popen(f'start "{name} (Port: {port})" cmd /K {command}', shell=True)
            elif sys.platform == "darwin":
                # macOS: osascript
                # Startet ein neues Terminal-Fenster und führt den Befehl darin aus.
                osa_command = f'tell app "Terminal" to do script "{command}"'
                subprocess.Popen(['osascript', '-e', osa_command])
            else:
                # Linux (versucht es mit gnome-terminal, was weit verbreitet ist)
                # Fällt auf xterm zurück, falls gnome-terminal nicht gefunden wird.
                try:
                    subprocess.Popen(['gnome-terminal', '--title', f'{name} (Port: {port})', '--', 'bash', '-c', f'{command}; exec bash'])
                except FileNotFoundError:
                    logging.warning("gnome-terminal nicht gefunden. Versuche es mit xterm.")
                    try:
                        subprocess.Popen(['xterm', '-T', f'{name} (Port: {port})', '-e', command])
                    except FileNotFoundError:
                        logging.error("Weder gnome-terminal noch xterm gefunden. Bitte starten Sie die Dienste manuell.")
                        return

            logging.info(f"'{name}' wird auf Port {port} gestartet.")
            time.sleep(1) # Kurze Pause, um zu vermeiden, dass alle Fenster gleichzeitig aufpoppen

        except Exception as e:
            logging.error(f"Fehler beim Starten von '{name}': {e}", exc_info=True)
            
    logging.info("\nAlle Dienste wurden gestartet. Bitte warten Sie einen Moment, bis alle vollständig initialisiert sind.")
    logging.info("Sie können diese jetzt in ihren jeweiligen Fenstern sehen und verwalten.")

def stop_services(services):
    """Findet und beendet alle Dienste, die auf den konfigurierten Ports laufen."""
    logging.info("Stoppe alle laufenden Microservices...")
    
    try:
        import psutil
    except ImportError:
        logging.error("Das 'psutil'-Paket wird benötigt, um Dienste zu stoppen. Bitte installieren Sie es mit: pip install psutil")
        return

    ports_to_kill = {service["port"] for service in services}
    processes_killed = 0

    # Iteriere über alle laufenden Prozesse
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Hole die Netzwerkverbindungen für jeden Prozess.
            # Die Methode net_connections() ersetzt die veraltete connections().
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port in ports_to_kill and conn.status == psutil.CONN_LISTEN:
                    process_name = proc.info['name']
                    pid = proc.info['pid']
                    logging.info(f"Stoppe Prozess '{process_name}' (PID: {pid}) auf Port {conn.laddr.port}...")
                    
                    try:
                        p = psutil.Process(pid)
                        p.terminate() # Sendet SIGTERM (sauberes Beenden)
                        p.wait(timeout=3) # Warte maximal 3 Sekunden
                        logging.info(f"Prozess {pid} erfolgreich beendet.")
                        processes_killed += 1
                    except psutil.NoSuchProcess:
                        logging.warning(f"Prozess {pid} wurde bereits beendet.")
                    except psutil.TimeoutExpired:
                        logging.warning(f"Prozess {pid} reagiert nicht auf 'terminate'. Erzwinge Beendigung (kill)...")
                        p.kill() # Sendet SIGKILL (erzwungenes Beenden)
                        p.wait()
                        logging.info(f"Prozess {pid} erzwungen beendet.")
                        processes_killed += 1
                        
                    ports_to_kill.remove(conn.laddr.port) # Port wurde behandelt
                    # Breche die innere Schleife ab, da der Prozess für diesen Port gefunden wurde
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass # Prozess existiert nicht mehr, wir haben keine Rechte oder es ist ein Zombie-Prozess
            
    if processes_killed > 0:
        logging.info(f"Insgesamt {processes_killed} Dienste erfolgreich gestoppt.")
    else:
        logging.info("Keine laufenden Dienste auf den konfigurierten Ports gefunden.")


def main():
    """Hauptfunktion, die die Kommandozeilenargumente verarbeitet."""
    parser = argparse.ArgumentParser(
        description="Ein Skript zum Verwalten der RAG-Microservices.",
        epilog="Beispiel: python manage_services.py start"
    )
    parser.add_argument(
        "action",
        choices=["start", "stop"],
        help="Die auszuführende Aktion: 'start' zum Starten aller Dienste, 'stop' zum Beenden."
    )
    
    args = parser.parse_args()
    
    services_config = load_config()
    
    if args.action == "start":
        start_services(services_config)
    elif args.action == "stop":
        stop_services(services_config)

if __name__ == "__main__":
    main()
