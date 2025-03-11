"""
Utility functions for UI components
"""
import tkinter as tk
from tkinter import ttk

def configure_dropdown_styles(root):
    """
    Configure dropdown listbox styles for the entire application
    
    This ensures that dropdown listboxes with black backgrounds have white text
    for better readability across all UI components.
    
    Args:
        root: The root/master window of the application
    """
    style = ttk.Style()
    
    # Configure the dropdown listbox styling
    style.map('TCombobox', 
             fieldbackground=[('readonly', 'white')],
             selectbackground=[('readonly', '#0078d7')],
             selectforeground=[('readonly', 'white')])
    
    # Apply these settings to the root window to ensure all comboboxes are affected
    root.option_add('*TCombobox*Listbox.background', 'black')
    root.option_add('*TCombobox*Listbox.foreground', 'white')
    root.option_add('*TCombobox*Listbox.selectBackground', '#0078d7')
    root.option_add('*TCombobox*Listbox.selectForeground', 'white')
    
    # Force more direct application of styles - these will take precedence
    root.tk.eval("""
        option add *TCombobox*Listbox.background black
        option add *TCombobox*Listbox.foreground white
        option add *TCombobox*Listbox.selectBackground #0078d7
        option add *TCombobox*Listbox.selectForeground white
        
        # Ensure these styles are applied immediately
        set my_style [ttk::style configure .]
        ttk::style map TCombobox -fieldbackground {readonly white} -selectbackground {readonly #0078d7} -selectforeground {readonly white}
    """)
    
    # Return the style object in case further customization is needed
    return style

def apply_listbox_style(listbox, bg_color="black", fg_color="white", select_bg="#0078d7", select_fg="white"):
    """
    Apply consistent styling to a listbox
    
    Args:
        listbox: The listbox widget to style
        bg_color: Background color
        fg_color: Text color
        select_bg: Selection background color
        select_fg: Selection text color
    """
    listbox.configure(
        bg=bg_color,
        fg=fg_color,
        selectbackground=select_bg,
        selectforeground=select_fg,
        borderwidth=1,
        highlightthickness=0
    )
    
    return listbox

def ensure_combobox_visible_text(combobox):
    """
    Ensure that a combobox's dropdown has visible text
    
    Args:
        combobox: A ttk.Combobox widget to configure
    """
    # This uses a more direct approach to force the listbox style
    combobox.tk.eval(f'''
    catch {{
        set popdown [ttk::combobox::PopdownWindow {combobox}]
        set listbox [winfo children $popdown]
        $listbox configure -background black -foreground white -selectbackground #0078d7 -selectforeground white
    }}
    ''')
    
    # Additionally, bind to the combobox opening event to reapply style
    combobox.bind("<<ComboboxDropdown>>", lambda event: _reapply_combobox_style(event.widget))
    
def _reapply_combobox_style(combobox):
    """Helper to reapply style when combobox is opened"""
    combobox.after(10, lambda: 
        combobox.tk.eval(f'''
        catch {{
            set popdown [ttk::combobox::PopdownWindow {combobox}]
            set listbox [winfo children $popdown]
            $listbox configure -background black -foreground white -selectbackground #0078d7 -selectforeground white
        }}
        ''')
    ) 