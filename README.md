# 📝 Handwritten Recognition API

🚀 A **FastAPI-based** application for **handwritten recognition and image cleaning.**  
This repository provides a structured and scalable API setup.

---

## 📂 Folder Structure
A well-organized folder structure ensures maintainability and scalability.

| Directory / File           | Description |
|---------------------------|------------|
| `app/`                    | Main application folder |
| ├── `main.py`             | Initializes the FastAPI app |
| ├── `config.py`           | Configuration settings (environment variables) |
| ├── `routes/`             | API route handlers |
| │   ├── `items.py`        | Routes related to items |
| │   ├── `users.py`        | Routes related to users |
| ├── `models/`             | Database models |
| │   ├── `item.py`         | Defines Item model |
| │   ├── `user.py`         | Defines User model |
| ├── `utils/`              | Helper functions |
| │   ├── `authentication.py` | Handles authentication logic |
| │   ├── `validation.py`    | Validation helpers |
| `tests/`                  | Unit & integration tests |
| ├── `routes/`             | Tests for API endpoints |
| ├── `models/`             | Tests for database models |
| ├── `utils/`              | Tests for utility functions |
| `.gitignore`              | Files to exclude from Git |
| `requirements.txt`        | Dependencies for the project |
| `README.md`               | Documentation |

---

## ⚙️ Developer Workflow
Maintaining a structured development workflow ensures smooth collaboration.

### 🏗️ **Branching Strategy**
- **Base Branch:** `dev` (always pull from `dev` before creating a new branch)
- **Branch Naming Conventions:**
  - `feature/<name>` → For new features (e.g., `feature/auth-system`)
  - `fix/<name>` → For bug fixes (e.g., `fix/user-login`)
  - `hotfix/<name>` → For urgent fixes on production
  - `refactor/<name>` → For code restructuring without new features

### 🔄 **Git Workflow**
| Action            | Description |
|------------------|-------------|
| **1️⃣ Pull Updates** | `git pull origin dev` (Keep local `dev` branch updated) |
| **2️⃣ Create Feature Branch** | `git checkout -b feature/<name>` |
| **3️⃣ Code & Commit** | Write code, then `git commit -m "Add: <feature description>"` |
| **4️⃣ Push Changes** | `git push origin feature/<name>` |
| **5️⃣ Open PR** | Open a Pull Request (PR) to merge into `dev` |
| **6️⃣ Code Review & Merge** | PR is reviewed and merged into `dev` |
| **7️⃣ Deploy** | Once tested, merge `dev` into `main` |

---

## 🛠️ Core Technologies
This project is built using the following stack:  
*(🔄 This list will be updated as more technologies are decided.)*

| Technology  | Usage |
|------------|------------|
| **FastAPI** | Web framework for building APIs |
| **Pydantic** | Data validation and settings |
| **Pillow (PIL)** | Image handling and manipulation | 
| **Scikit-learn** | Feature extraction and image processing utilities |
| **TensorFlow / PyTorch** | Deep learning models for handwritten text recognition |

---

## 🔖 Best Practices 🚀 
 
<summary>✅ Always create feature branches from `dev`</summary>  
<summary>✅ Use clear commit messages</summary>  
<summary>✅ Add: User authentication system</summary>  
<summary>✅ Pull the latest changes before pushing</summary>  
<summary>✅ git pull origin dev</summary>  
<summary>✅ Write unit tests for all new features</summary>  
<summary>✅ Code reviews before merging PRs</summary>  
<summary>✅ git pull origin dev</summary>  