#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Checking Script for Synthetic Data Generation App
This script performs various checks on the application code.
"""

import sys
import ast
import importlib.util
from pathlib import Path
import subprocess
import os

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Color codes for terminal output (with ASCII fallbacks)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ASCII-safe symbols
SUCCESS_SYMBOL = '[OK]'
ERROR_SYMBOL = '[X]'
WARNING_SYMBOL = '[!]'

def print_header(text):
    """Print a formatted header."""
    try:
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    except:
        print(f"\n{'='*60}")
        print(f"{text}")
        print(f"{'='*60}\n")

def print_success(text):
    """Print success message."""
    try:
        print(f"{Colors.GREEN}{SUCCESS_SYMBOL} {text}{Colors.RESET}")
    except:
        print(f"{SUCCESS_SYMBOL} {text}")

def print_error(text):
    """Print error message."""
    try:
        print(f"{Colors.RED}{ERROR_SYMBOL} {text}{Colors.RESET}")
    except:
        print(f"{ERROR_SYMBOL} {text}")

def print_warning(text):
    """Print warning message."""
    try:
        print(f"{Colors.YELLOW}{WARNING_SYMBOL} {text}{Colors.RESET}")
    except:
        print(f"{WARNING_SYMBOL} {text}")

def check_syntax(file_path):
    """Check Python syntax errors."""
    print_header("Checking Syntax")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print_success(f"Syntax check passed for {file_path}")
        return True
    except SyntaxError as e:
        print_error(f"Syntax error in {file_path}:")
        print(f"  Line {e.lineno}: {e.text}")
        print(f"  Error: {e.msg}")
        return False
    except Exception as e:
        print_error(f"Error reading {file_path}: {str(e)}")
        return False

def check_imports(file_path):
    """Check if all imports are available."""
    print_header("Checking Imports")
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Required packages
        required_packages = {
            'streamlit': 'streamlit',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'sklearn': 'scikit-learn',
        }
        
        for imp in imports:
            if imp in required_packages:
                try:
                    __import__(imp)
                    print_success(f"Import '{imp}' is available")
                except ImportError:
                    issues.append(f"Missing package: {required_packages[imp]}")
                    print_error(f"Import '{imp}' not available. Install: pip install {required_packages[imp]}")
        
        if not issues:
            print_success("All required imports are available")
            return True
        else:
            return False
            
    except Exception as e:
        print_error(f"Error checking imports: {str(e)}")
        return False

def check_streamlit_specific(file_path):
    """Check for Streamlit-specific issues."""
    print_header("Checking Streamlit-Specific Code")
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check for common Streamlit patterns
        checks = {
            'st.set_page_config': 'Page config should be set early',
            'st.session_state': 'Session state usage',
            'st.sidebar': 'Sidebar usage',
            'st.tabs': 'Tabs usage',
        }
        
        for pattern, description in checks.items():
            if pattern in code:
                print_success(f"Found {description}")
            else:
                if pattern == 'st.set_page_config':
                    print_warning(f"Consider adding {pattern} for better page configuration")
        
        # Check for potential issues
        if 'st.cache' in code or '@st.cache' in code:
            print_warning("Using deprecated st.cache. Consider using st.cache_data or st.cache_resource")
        
        # Check for proper error handling
        if 'try:' in code and 'except' in code:
            print_success("Error handling found in code")
        else:
            print_warning("Consider adding error handling for better user experience")
        
        return True
        
    except Exception as e:
        print_error(f"Error checking Streamlit code: {str(e)}")
        return False

def check_file_structure():
    """Check if required files exist."""
    print_header("Checking File Structure")
    required_files = {
        'app.py': 'Main application file',
        'requirements.txt': 'Dependencies file',
        'README.md': 'Documentation file',
    }
    
    all_exist = True
    for file, description in required_files.items():
        if Path(file).exists():
            print_success(f"{file} exists ({description})")
        else:
            print_error(f"{file} is missing ({description})")
            all_exist = False
    
    return all_exist

def check_requirements():
    """Check requirements.txt format."""
    print_header("Checking requirements.txt")
    
    if not Path('requirements.txt').exists():
        print_warning("requirements.txt not found")
        return False
    
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        valid_packages = []
        issues = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Basic validation
            if '>=' in line or '==' in line or not any(c in line for c in ['=', '<', '>']):
                valid_packages.append(line)
                print_success(f"Valid package specification: {line}")
            else:
                issues.append(f"Line {i}: {line}")
                print_warning(f"Line {i} may have formatting issues: {line}")
        
        if not issues:
            print_success("requirements.txt format is valid")
            return True
        else:
            return False
            
    except Exception as e:
        print_error(f"Error reading requirements.txt: {str(e)}")
        return False

def run_linter(file_path):
    """Run pylint or flake8 if available."""
    print_header("Running Linter Checks")
    
    linters = {
        'pylint': ['pylint', '--disable=all', '--enable=E,F', file_path],
        'flake8': ['flake8', '--select=E,F', file_path],
    }
    
    linter_found = False
    for linter_name, command in linters.items():
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print_success(f"{linter_name} found no critical errors")
                linter_found = True
            else:
                print_warning(f"{linter_name} found some issues:")
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                linter_found = True
        except FileNotFoundError:
            print_warning(f"{linter_name} not installed. Install with: pip install {linter_name}")
        except subprocess.TimeoutExpired:
            print_warning(f"{linter_name} timed out")
        except Exception as e:
            print_warning(f"Error running {linter_name}: {str(e)}")
    
    if not linter_found:
        print_warning("No linters available. Install pylint or flake8 for additional checks.")
    
    return True

def check_code_quality(file_path):
    """Check for common code quality issues."""
    print_header("Checking Code Quality")
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check for long lines
        long_lines = []
        for i, line in enumerate(lines, 1):
            if len(line.rstrip()) > 120:
                long_lines.append((i, len(line.rstrip())))
        
        if long_lines:
            print_warning(f"Found {len(long_lines)} lines longer than 120 characters")
            for line_num, length in long_lines[:5]:  # Show first 5
                print(f"  Line {line_num}: {length} characters")
        else:
            print_success("No excessively long lines found")
        
        # Check for TODO/FIXME comments
        todos = []
        for i, line in enumerate(lines, 1):
            if 'TODO' in line.upper() or 'FIXME' in line.upper():
                todos.append((i, line.strip()[:60]))
        
        if todos:
            print_warning(f"Found {len(todos)} TODO/FIXME comments")
            for line_num, comment in todos[:5]:
                print(f"  Line {line_num}: {comment}...")
        else:
            print_success("No TODO/FIXME comments found")
        
        # Check for print statements (should use st.write in Streamlit)
        print_statements = []
        for i, line in enumerate(lines, 1):
            if 'print(' in line and 'st.' not in line:
                print_statements.append(i)
        
        if print_statements:
            print_warning(f"Found {len(print_statements)} print() statements (consider using st.write for Streamlit)")
        else:
            print_success("No print() statements found (good for Streamlit)")
        
        return len(issues) == 0
        
    except Exception as e:
        print_error(f"Error checking code quality: {str(e)}")
        return False

def main():
    """Main function to run all checks."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("  Synthetic Data Generation App - Error Checker")
    print("=" * 60)
    print(f"{Colors.RESET}\n")
    
    app_file = Path('app.py')
    
    if not app_file.exists():
        print_error("app.py not found in current directory")
        sys.exit(1)
    
    results = {
        'syntax': check_syntax(app_file),
        'imports': check_imports(app_file),
        'streamlit': check_streamlit_specific(app_file),
        'structure': check_file_structure(),
        'requirements': check_requirements(),
        'quality': check_code_quality(app_file),
    }
    
    # Run linter (non-blocking)
    run_linter(app_file)
    
    # Summary
    print_header("Summary")
    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    for check_name, passed in results.items():
        try:
            status = f"{Colors.GREEN}PASSED{Colors.RESET}" if passed else f"{Colors.RED}FAILED{Colors.RESET}"
            print(f"  {check_name.upper():<20} {status}")
        except:
            status = "PASSED" if passed else "FAILED"
            print(f"  {check_name.upper():<20} {status}")
    
    try:
        print(f"\n{Colors.BOLD}Total: {passed_checks}/{total_checks} checks passed{Colors.RESET}\n")
    except:
        print(f"\nTotal: {passed_checks}/{total_checks} checks passed\n")
    
    if passed_checks == total_checks:
        try:
            print(f"{Colors.GREEN}{Colors.BOLD}All checks passed!{Colors.RESET}\n")
        except:
            print(f"All checks passed!\n")
        sys.exit(0)
    else:
        try:
            print(f"{Colors.YELLOW}{Colors.BOLD}Some checks failed. Please review the issues above.{Colors.RESET}\n")
        except:
            print(f"Some checks failed. Please review the issues above.\n")
        sys.exit(1)

if __name__ == '__main__':
    main()

