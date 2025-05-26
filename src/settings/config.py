import json
import os
import copy

# Path to the configuration file
CONFIG_FILE_PATH = "settings.json"

# Default settings structure
DEFAULT_SETTINGS = {
    "wakeword_settings": {
        "model_path1": "",
        "model_path2": "",
        "model_path3": "",
        "chunk_size": 1280,
        "inference_framework": "onnx",
        "silence_threshold": 500,
        "silence_duration_seconds": 1.0,
        "no_speech_timeout_seconds": 4.0
    }
}

def save_settings(settings, filepath=CONFIG_FILE_PATH):
    """
    Saves the settings dictionary to a JSON file.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Settings saved to {filepath}")
    except IOError as e:
        print(f"Error saving settings to {filepath}: {e}")

def load_settings(filepath=CONFIG_FILE_PATH):
    """
    Loads settings from a JSON file, merges with defaults, and cleans up unknown keys.
    If the file does not exist or is invalid, creates it with default settings.
    """
    if not os.path.exists(filepath):
        print(f"Settings file not found at {filepath}. Creating with default settings.")
        save_settings(DEFAULT_SETTINGS, filepath)
        return copy.deepcopy(DEFAULT_SETTINGS)

    try:
        with open(filepath, 'r') as f:
            loaded_from_file = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading or parsing {filepath}: {e}. Using default settings and overwriting.")
        save_settings(DEFAULT_SETTINGS, filepath)
        return copy.deepcopy(DEFAULT_SETTINGS)

    # Start with a deep copy of the defaults
    final_settings = copy.deepcopy(DEFAULT_SETTINGS)

    # Merge loaded settings with defaults
    for category_key, default_category_values in DEFAULT_SETTINGS.items():
        if category_key in loaded_from_file:
            loaded_category_values = loaded_from_file[category_key]
            if isinstance(default_category_values, dict) and isinstance(loaded_category_values, dict):
                # Merge each key in the category
                current_category_settings = final_settings[category_key]
                for key, _ in default_category_values.items():
                    if key in loaded_category_values:
                        current_category_settings[key] = loaded_category_values[key]
            elif not isinstance(default_category_values, dict) and category_key in loaded_from_file:
                final_settings[category_key] = loaded_from_file[category_key]

    # Remove unknown keys from loaded settings (for comparison)
    cleaned_settings_for_comparison = copy.deepcopy(final_settings)
    for category_key, default_category_values in DEFAULT_SETTINGS.items():
        if isinstance(default_category_values, dict) and \
           category_key in cleaned_settings_for_comparison and \
           isinstance(cleaned_settings_for_comparison[category_key], dict):
            final_category = cleaned_settings_for_comparison[category_key]
            keys_to_remove = [k for k in final_category if k not in default_category_values]
            for k in keys_to_remove:
                del final_category[k]

    # Ensure all default keys exist and remove unknown keys from final settings
    for category_key, default_category_values in DEFAULT_SETTINGS.items():
        if isinstance(default_category_values, dict) and \
           category_key in final_settings and \
           isinstance(final_settings[category_key], dict):
            current_category_in_final = final_settings[category_key]
            # Add missing keys
            for default_key, default_val in default_category_values.items():
                if default_key not in current_category_in_final:
                    current_category_in_final[default_key] = default_val
            # Remove unknown keys
            keys_to_remove_from_final = [k for k in current_category_in_final if k not in default_category_values]
            for k_remove in keys_to_remove_from_final:
                del current_category_in_final[k_remove]
        elif category_key not in final_settings and isinstance(default_category_values, dict):
            final_settings[category_key] = copy.deepcopy(default_category_values)

    # Save cleaned settings if they differ from loaded file
    if final_settings != loaded_from_file:
        save_settings(final_settings, filepath)

    return final_settings

def clear_screen():
    """
    Clears the terminal screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def edit_settings_menu(settings_dict, settings_category_key, default_category_settings):
    """
    Manages editing for a specific category of settings.
    """
    model_path_keys = ["model_path1", "model_path2", "model_path3"]

    while True:
        clear_screen()
        print(f"===== Edit {settings_category_key.replace('_', ' ').title()} =====")

        current_category_settings = settings_dict[settings_category_key]
        keys = sorted(list(current_category_settings.keys()))

        # Display current settings
        for i, key in enumerate(keys):
            value_display = current_category_settings[key]
            if key in model_path_keys:
                if not value_display:
                    value_display = "None"
            print(f"  {i+1}. {key:25}: {value_display}")

        print(f"\n  {len(keys)+1}. Back to Main Menu")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(keys):
                index = choice_num - 1
                key_to_change = keys[index]

                # Special handling for model path keys
                if key_to_change in model_path_keys and settings_category_key == "wakeword_settings":
                    clear_screen()
                    print(f"===== Select Model for {key_to_change} =====")
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    custom_model_dir = os.path.join(project_root, "src", "wakeword", "custom")

                    # Ensure custom model directory exists
                    if not os.path.exists(custom_model_dir):
                        try:
                            os.makedirs(custom_model_dir)
                            print(f"Created directory for custom models: {custom_model_dir}")
                        except OSError as e:
                            print(f"Error creating directory {custom_model_dir}: {e}")
                            print("Please create this directory manually and place your .onnx models there.")
                            input("Press Enter to continue...")
                            continue

                    print(f"\nINFO: Place your .onnx wakeword model files in:\n  '{custom_model_dir}'\n")

                    available_model_filenames = []
                    try:
                        if os.path.isdir(custom_model_dir):
                            available_model_filenames = sorted([f for f in os.listdir(custom_model_dir) if f.endswith(".onnx")])
                        else:
                            print(f"Error: {custom_model_dir} is not a directory.")
                    except OSError as e:
                        print(f"Error reading models from {custom_model_dir}: {e}")

                    print("Available options:")
                    print("  0. None (clear path)")
                    if not available_model_filenames:
                        print("     (No .onnx models found in the custom directory)")
                    else:
                        for i, model_name in enumerate(available_model_filenames):
                            print(f"  {i+1}. {model_name} (uses this filename from custom folder)")

                    manual_path_option_num = len(available_model_filenames) + 1
                    print(f"  {manual_path_option_num}. Enter Full Path Manually")

                    current_value = current_category_settings.get(key_to_change, "")
                    display_current = current_value if current_value else "None"
                    if current_value and not os.path.isabs(current_value) and os.path.exists(os.path.join(custom_model_dir, current_value)):
                        display_current = f"{current_value} (from custom folder)"
                    elif os.path.isabs(current_value):
                        display_current = f"{current_value} (full path)"

                    print(f"\nCurrently selected for {key_to_change}: {display_current}")

                    model_choice_str = input("Enter number of your choice (or press Enter to cancel): ").strip()

                    if not model_choice_str:
                        print("Selection cancelled.")
                        input("Press Enter to return...")
                        continue

                    try:
                        model_choice_num = int(model_choice_str)
                        new_model_path_value = current_value

                        if model_choice_num == 0:
                            new_model_path_value = ""
                            print("Path cleared.")
                        elif 1 <= model_choice_num <= len(available_model_filenames):
                            new_model_path_value = available_model_filenames[model_choice_num - 1]
                            print(f"Selected model filename: {new_model_path_value}")
                        elif model_choice_num == manual_path_option_num:
                            manual_path = input("Enter the full path to the .onnx model file: ").strip()
                            if manual_path:
                                new_model_path_value = manual_path
                                print(f"Full path set: {new_model_path_value}")
                            else:
                                print("Manual path entry cancelled. No change made.")
                                new_model_path_value = current_value
                        else:
                            print("Invalid option number. No change made.")
                            new_model_path_value = current_value

                        settings_dict[settings_category_key][key_to_change] = new_model_path_value
                        save_settings(settings_dict)
                        if new_model_path_value != current_value:
                            print("Settings saved.")
                    except ValueError:
                        print("Invalid input. Please enter a number. No change made.")
                    input("Press Enter to return to Wakeword Settings...")
                    continue

                # For other keys, prompt for new value
                current_value = current_category_settings[key_to_change]
                clear_screen()
                print(f"===== Edit Setting: {key_to_change} ({settings_category_key.replace('_', ' ').title()}) =====")
                print(f"Current value: {current_value}")

                new_value_str = input(f"Enter new value for '{key_to_change}' (or press Enter to cancel): ").strip()

                if new_value_str == "":
                    print("Change cancelled.")
                    continue

                original_type = type(default_category_settings.get(key_to_change, ""))

                try:
                    if new_value_str.lower() == '""' and original_type is str:
                        new_value = ""
                    elif original_type is int:
                        new_value = int(new_value_str)
                    elif original_type is float:
                        new_value = float(new_value_str)
                    elif original_type is str:
                        new_value = new_value_str
                    else:
                        print(f"Unsupported type for key '{key_to_change}'. Treating as string.")
                        new_value = new_value_str

                    settings_dict[settings_category_key][key_to_change] = new_value
                    save_settings(settings_dict)
                    print(f"'{key_to_change}' updated to '{new_value}' and settings saved.")
                except ValueError:
                    print(f"Invalid value format for type {original_type.__name__}. Setting not changed.")
            elif choice_num == len(keys) + 1:
                # Back to main menu
                break
            else:
                print("Invalid number. Please try again.")
        else:
            print("Invalid choice. Please try again.")

def open_settings():
    """
    Main settings menu loop.
    """
    settings = load_settings()

    while True:
        clear_screen()
        print("===== Main Settings Menu =====")
        print("  1. Wakeword Settings")
        print("\n  save. Save and Exit")
        print("  exit. Exit")

        main_choice = input("\nEnter your choice: ").strip().lower()

        if main_choice == '1':
            edit_settings_menu(settings, "wakeword_settings", DEFAULT_SETTINGS["wakeword_settings"])
        elif main_choice == 'save':
            save_settings(settings)
            print("Settings saved. Exiting.")
            break
        elif main_choice == 'exit':
            print("Exiting settings editor.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    open_settings()
