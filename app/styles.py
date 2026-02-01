# app/styles.py

def get_theme_css(bg_color, sidebar_color, accent_color, text_color, component_bg, component_text_color):
    """
    Returns CSS string for a given theme configuration.
    
    Args:
        bg_color: Main page background
        sidebar_color: Sidebar background
        accent_color: Used for Borders, Buttons, Table Headers, Dropdown Backgrounds
        text_color: General page text color
        component_bg: Background for standard input boxes (like file uploader)
        component_text_color: Text color inside dropdowns and tags (usually white for contrast)
    """
    return f"""
    <style>
    /* 1. Main Background */
    .stApp {{
        background-color: {bg_color};
    }}
    
    /* 2. Sidebar Background */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_color};
    }}
    
    /* 3. Hide Top Header Bar */
    header[data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}
    
    /* 4. General Text Color */
    h1, h2, h3, h4, h5, h6, p, label, li, span {{
        color: {text_color} !important;
    }}
    
    /* 5. BUTTONS - Make button text white */
    div.stButton > button {{
        background-color: {accent_color};
        color: {component_text_color} !important;  /* Changed from white to use theme's component_text_color */
        border: none;
        font-weight: bold;
    }}
    div.stButton > button:hover {{
        border: 1px solid {text_color};
        background-color: {text_color};
        color: {component_text_color} !important;
    }}
    /* Force white text on button hover state too */
    div.stButton > button:hover p,
    div.stButton > button:hover div,
    div.stButton > button:hover span {{
        color: {component_text_color} !important;
    }}

    /* 6. DROPDOWNS (Selectbox) - Green BG / White Text */
    div[data-baseweb="select"] > div {{
        background-color: {accent_color} !important;
        border-color: {accent_color};
        color: {component_text_color} !important;
    }}
    /* Dropdown Text & Icons */
    div[data-baseweb="select"] span {{
        color: {component_text_color} !important;
    }}
    div[data-baseweb="select"] svg {{
        fill: {component_text_color} !important;
    }}
    
    /* 6b. DROPDOWN OPTIONS LIST - White text on dark background */
    ul[data-baseweb="menu"] li, 
    ul[data-baseweb="menu"] li div,
    ul[data-baseweb="menu"] li span {{
        color: {component_text_color} !important;
        background-color: {accent_color} !important;
    }}
    ul[data-baseweb="menu"] li:hover {{
        background-color: {text_color} !important;
    }}

    /* 7. MULTISELECT TAGS (Remove default Red) */
    span[data-baseweb="tag"] {{
        background-color: {accent_color} !important;
        color: {component_text_color} !important;
    }}
    span[data-baseweb="tag"] span {{
        color: {component_text_color} !important;
    }}
    
    /* 7b. MULTISELECT DROPDOWN OPTIONS */
    div[data-testid="stMultiSelect"] ul[data-baseweb="menu"] li,
    div[data-testid="stMultiSelect"] ul[data-baseweb="menu"] li div,
    div[data-testid="stMultiSelect"] ul[data-baseweb="menu"] li span {{
        color: {component_text_color} !important;
        background-color: {accent_color} !important;
    }}

    /* 8. File Uploader Box */
    [data-testid="stFileUploader"] section {{
        background-color: {component_bg};
        border: 2px dashed {accent_color};
    }}
    
    /* 9. HTML Table Styling (st.table) */
    table {{
        color: {text_color};
        background-color: {component_bg};
        border-collapse: collapse;
        border: 1px solid {accent_color};
        width: 100%;
    }}
    /* TABLE HEADERS - Force white text on accent color background */
    th {{
        background-color: {accent_color} !important;
        color: {component_text_color} !important;
        border-bottom: 2px solid {component_text_color};
        font-weight: bold;
    }}
    /* Override any child elements in table headers */
    th span, th div, th p {{
        color: {component_text_color} !important;
    }}
    td {{
        border-bottom: 1px solid {sidebar_color};
        color: {text_color};
    }}
    
    /* 10. SPECIFIC OVERRIDES for dropdown labels */
    /* For the selectbox widget container */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stColumn"] label {{
        color: {text_color} !important;
        font-weight: bold;
    }}
    
    /* 11. Dropdown option list container */
    div[role="listbox"] {{
        background-color: {accent_color} !important;
    }}
    div[role="option"] {{
        color: {component_text_color} !important;
    }}
    
    /* 12. Override for Streamlit's default text colors in dropdowns */
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stMultiSelect span[data-baseweb="tag"],
    .stMultiSelect div[data-baseweb="popover"] {{
        color: {component_text_color} !important;
    }}
    
    /* 13. Popover (dropdown list) styling */
    div[data-baseweb="popover"] {{
        background-color: {accent_color} !important;
        border: 1px solid {accent_color} !important;
    }}
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li div,
    div[data-baseweb="popover"] li span {{
        color: {component_text_color} !important;
    }}
    
    /* 14. SPECIFIC BUTTON TEXT OVERRIDES */
    /* Target the button text directly */
    button[data-testid="baseButton-secondary"] p,
    button[data-testid="baseButton-secondary"] div,
    button[data-testid="baseButton-secondary"] span,
    .stButton button p,
    .stButton button div,
    .stButton button span {{
        color: {component_text_color} !important;
    }}
    
    /* 15. Ensure all buttons have white text */
    button {{
        color: {component_text_color} !important;
    }}
    button p, button div, button span {{
        color: {component_text_color} !important;
    }}
    </style>
    """

# Predefined themes for different pages
class Themes:
    @staticmethod
    def upload():
        """Blue theme for Upload page"""
        return {
            "bg_color": "#E6F3FF",
            "sidebar_color": "#CCE5FF",
            "accent_color": "#0052CC",
            "text_color": "#002244",
            "component_bg": "#FFFFFF",
            "component_text_color": "#FFFFFF"
        }
    
    @staticmethod
    def analytics():
        """Green theme for Analytics page"""
        return {
            "bg_color": "#F1F8E9",
            "sidebar_color": "#DCEDC8",
            "accent_color": "#33691E",
            "text_color": "#1B5E20",
            "component_bg": "#FFFFFF",
            "component_text_color": "#FFFFFF"  # White text for buttons and dropdowns
        }
    
    @staticmethod
    def directory():
        """Neutral theme for Directory page"""
        return {
            "bg_color": "#FFFFFF",
            "sidebar_color": "#9D9191",
            "accent_color": "#343A40",
            "text_color": "#000000",
            "component_bg": "#FFFFFF",
            "component_text_color": "#FFFFFF"
        }
    
    @staticmethod
    def get_css(theme_name):
        """Get CSS string for a theme name"""
        themes = {
            "upload": Themes.upload(),
            "analytics": Themes.analytics(),
            "directory": Themes.directory(),
        }
        if theme_name in themes:
            return get_theme_css(**themes[theme_name])
        return ""