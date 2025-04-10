import streamlit as st

# Page Config
st.set_page_config(
    page_title="Contact | BrainAI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .contact-box {
            background-color: #f5f9ff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 2px 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        .member-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.04);
            margin-bottom: 25px;
        }
        .member-card img {
            border-radius: 50%;
        }
        .centered {
            text-align: center;
        }
        .github-icon {
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown("<h2 style='color:#0d47a1;'>Contact Us</h2>", unsafe_allow_html=True)

# Contact Info Section
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.markdown("""
    <div class="contact-box">
        <h4>Get In Touch</h4>
        <p>If you have any questions or feedback regarding our Brain Tumor Detection System, please feel free to reach out:</p>
        <ul>
            <li><strong>Email</strong>: contact@braintumorai.com</li>
            <li><strong>Phone</strong>: +1 (555) 123–4567</li>
            <li><strong>Address</strong>: 123 Medical AI Drive, Boston, MA</li>
        </ul>
        <h5>Support</h5>
        <p>Please include:</p>
        <ul>
            <li>Your full name and contact details</li>
            <li>Detailed description of the issue</li>
            <li>Attachment/screenshots if applicable</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.empty()

# Divider
st.markdown("---")

# Project Guide Section
st.subheader("Project Guide")

guide_info = {
    "name": "Ms Priyanka H.L",
    "designation": "Assistant Professor, Dept. of Information Science and Engineering",
    "institute": "Malnad College of Engineering",
    "email": "phl@mcehassan.ac.in",
    "image": "https://www.mcehassan.ac.in/assets/departments/IS/faculty/profile_227-231501.jpg"
}

col_guide1, col_guide2 = st.columns([1, 3])
with col_guide1:
    st.image(guide_info["image"], width=120)
with col_guide2:
    st.markdown(f"""
    <div class="contact-box">
        <h4 style='color:#1565c0;'>{guide_info['name']}</h4>
        <p><i>{guide_info['designation']}</i></p>
        <p>{guide_info['institute']}<br>{guide_info['email']}</p>
    </div>
    """, unsafe_allow_html=True)

# Divider
st.markdown("---")

# Project Team Section
st.subheader("Project Team")

team_members = [
    {"name": "Aarav Patel", "email": "aarav@example.com", "github": "https://github.com/aaravpatel"},
    {"name": "Meera Sharma", "email": "meera@example.com", "github": "https://github.com/meerasharma"},
    {"name": "Rohan Verma", "email": "rohan@example.com", "github": "https://github.com/rohanverma"},
    {"name": "Anaya Singh", "email": "anaya@example.com", "github": "https://github.com/anayasingh"},
]

cols = st.columns(4)
for i, member in enumerate(team_members):
    with cols[i]:
        st.markdown(f"""
        <div class="member-card">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="70">
            <p style="margin: 10px 0 5px; font-weight: 600; font-size: 16px;">{member['name']}</p>
            <p style="font-size: 14px; color: #555;">{member['email']}</p>
            <a href="{member['github']}" target="_blank" class="github-icon">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733609.png" width="26" title="GitHub Profile">
            </a>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p class='centered' style='color: #757575;'>© 2025 BrainTumorAI | All rights reserved.</p>", unsafe_allow_html=True)
