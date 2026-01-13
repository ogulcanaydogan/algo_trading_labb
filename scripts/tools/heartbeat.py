#!/usr/bin/env python3
"""Background heartbeat to keep dashboard timestamps fresh."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

state_path = Path('data/state.json')
bot_state_path = Path('data/bot_state.json')

print('Heartbeat started. Updating timestamps every 5 seconds...')
print('Press Ctrl+C to stop')

try:
    while True:
        now = datetime.now(timezone.utc).isoformat()
        
        # Update state.json
        try:
            if state_path.exists():
                with state_path.open('r') as f:
                    state = json.load(f)
                state['timestamp'] = now
                with state_path.open('w') as f:
                    json.dump(state, f, indent=2)
        except Exception as e:
            print(f'Error updating state.json: {e}')
        
        # Update bot_state.json
        try:
            if bot_state_path.exists():
                with bot_state_path.open('r') as f:
                    bot_state = json.load(f)
                bot_state['timestamp'] = now
                bot_state['last_heartbeat'] = now
                with bot_state_path.open('w') as f:
                    json.dump(bot_state, f, indent=2)
        except Exception as e:
            print(f'Error updating bot_state.json: {e}')
        
        print(f'✓ {datetime.now().strftime("%H:%M:%S")}')
        time.sleep(5)
except KeyboardInterrupt:
    print('\nHeartbeat stopped.')
