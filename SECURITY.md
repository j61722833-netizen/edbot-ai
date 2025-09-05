# EdBot AI - Security Guide

## 🔐 API Key Security Best Practices

EdBot AI implements multiple layers of security for handling API keys and sensitive configuration data.

### 🎯 Security Principles

1. **Never commit API keys to version control**
2. **Use secure storage methods for local development**
3. **Validate API key formats before use**
4. **Provide clear error messages without exposing keys**
5. **Support multiple configuration methods for different environments**

## 🛠 Configuration Methods

### 1. System Keyring (Most Secure) ✅ Recommended

Store API keys in your operating system's secure keyring:

```bash
python setup_keys.py
# Choose option 1: System keyring for each API key
```

**Benefits:**
- Encrypted storage using OS security features
- No files with keys on disk
- Automatic unlocking with OS login
- Works across different terminals/sessions

**Supported platforms:**
- macOS: Keychain
- Windows: Windows Credential Locker  
- Linux: Secret Service (gnome-keyring, kwallet)

### 2. Environment Variables (Production) 🏭

For production deployments and CI/CD:

```bash
# Set environment variables
export OPENAI_API_KEY='sk-...'
export LANGSMITH_API_KEY='ls_...'

# Or in production environment
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls_...
```

### 3. .env File (Development) 📝

For local development convenience:

```bash
# Create .env file (automatically ignored by git)
python setup_keys.py
# Choose option 4: Save to .env file

# Or create manually:
echo "OPENAI_API_KEY=sk-..." > .env
```

## 🚨 Security Features

### Automatic Git Exclusion

The following files are automatically excluded from version control:

```gitignore
# Security - API Keys and Sensitive Data
.env
.env.*
!.env.example
*.key
*.pem
secrets/
config/local/
api_keys.json
credentials.json

# EdBot AI specific
vector_indexes/
outputs/
extracted_text/*.json
```

### API Key Validation

All API keys are validated for format before use:

- **OpenAI**: Must start with `sk-` and be at least 20 characters
- **Format checking** prevents accidental invalid keys
- **No logging** of API key values

### Secure Error Handling

Error messages never expose API key values:

```python
# ❌ Bad - exposes key
raise ValueError(f"Invalid API key: {api_key}")

# ✅ Good - secure message
raise ValueError("OpenAI API key required. Run: python setup_keys.py")
```

## 📋 Setup Guide

### Quick Setup

1. **Install dependencies:**
   ```bash
   pip install keyring  # For secure keyring support
   ```

2. **Run interactive setup:**
   ```bash
   python setup_keys.py
   ```

3. **Choose your security method:**
   - System keyring (most secure)
   - .env file (convenient)
   - Environment variables (production)

### Manual Setup Options

#### Option 1: Keyring Setup
```bash
python -c "
import keyring
keyring.set_password('edbot-ai', 'openai_api_key', 'your-key-here')
"
```

#### Option 2: Environment Variables
```bash
# Linux/Mac
export OPENAI_API_KEY='your-key-here'

# Windows
set OPENAI_API_KEY=your-key-here
```

#### Option 3: .env File
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
echo "LANGSMITH_TRACING=true" >> .env
```

## 🔍 Configuration Verification

Check your configuration status:

```bash
# Interactive status check
python setup_keys.py
# Choose option 5: Show current status

# Command line status
python -m src.config.settings --status
```

Expected output:
```json
{
  "configured": true,
  "openai_api_key_set": true,
  "langsmith_api_key_set": true,
  "config_sources": {
    "openai_api_key": "keyring",
    "langsmith_api_key": "environment"
  },
  "keyring_available": true
}
```

## 🚀 Production Deployment

### Environment Variables (Recommended)

Set environment variables in your production environment:

```bash
# Docker
docker run -e OPENAI_API_KEY=sk-... edbot-ai

# Kubernetes
apiVersion: v1
kind: Secret
metadata:
  name: edbot-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: sk-...
```

### Configuration Validation

The system automatically validates configuration on startup:

```bash
# Test configuration
python -c "
from src.config.settings import get_config
config = get_config()
print('✅ Configuration valid!' if config.is_configured() else '❌ Missing API keys')
"
```

## 🛡 Security Checklist

- [ ] ✅ API keys never committed to git
- [ ] ✅ .env file in .gitignore  
- [ ] ✅ API key format validation
- [ ] ✅ Secure keyring storage option
- [ ] ✅ Environment variable support
- [ ] ✅ No API keys in error messages
- [ ] ✅ No API keys in logs
- [ ] ✅ Multiple configuration methods
- [ ] ✅ Configuration validation
- [ ] ✅ Clear setup instructions

## 🚨 Emergency Procedures

### Compromised API Key

1. **Revoke the key immediately:**
   - OpenAI: https://platform.openai.com/api-keys
   - LangSmith: https://smith.langchain.com/settings

2. **Clear local storage:**
   ```bash
   # Clear keyring
   python setup_keys.py
   # Choose option 6: Clear keyring data
   
   # Remove .env file
   rm .env
   
   # Clear environment
   unset OPENAI_API_KEY
   ```

3. **Generate new key and reconfigure:**
   ```bash
   python setup_keys.py
   ```

### Audit Configuration

```bash
# Check all configuration sources
python -m src.config.settings --status

# Verify no keys in environment
env | grep -i api_key

# Check for any .env files
find . -name ".env*" -type f
```

## 📚 Additional Resources

- **OpenAI API Key Management**: https://platform.openai.com/api-keys
- **Python Keyring Documentation**: https://keyring.readthedocs.io/
- **Environment Variables Best Practices**: https://12factor.net/config
- **Git Security**: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure

## ⚠️ Important Notes

1. **Never share API keys** in chat, email, or documentation
2. **Regularly rotate API keys** (recommended: monthly)
3. **Monitor API usage** for unexpected activity
4. **Use minimum required permissions** for each API key
5. **Keep backup access methods** for key recovery

---

**Security is a shared responsibility. Follow these practices to keep your EdBot AI installation secure.** 🔐