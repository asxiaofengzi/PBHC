#!/bin/bash
# filepath: /Users/hahahaha/Desktop/c++/python/PBHC/innovation_multimodal/setup_multimodal.sh

set -e

echo "🎯 Setting up multimodal motion fusion learning system..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PBHC_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}📁 PBHC Root: $PBHC_ROOT${NC}"
echo -e "${BLUE}📁 Multimodal Dir: $SCRIPT_DIR${NC}"

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        exit 1
    fi
}

# Step 1: Check Python environment
echo -e "\n${YELLOW}🐍 Checking Python environment...${NC}"
python3 --version
print_status $? "Python version check"

# Step 2: Install dependencies
echo -e "\n${YELLOW}📦 Installing dependencies...${NC}"
pip3 install --upgrade pip
pip3 install hydra-core omegaconf wandb tensorboard matplotlib seaborn plotly scipy scikit-learn
print_status $? "Dependencies installation"

# Step 3: Run integration script
echo -e "\n${YELLOW}🔧 Running integration script...${NC}"
cd "$SCRIPT_DIR"
python3 integrate_multimodal.py
print_status $? "Integration script"

# Step 4: Create convenience scripts
echo -e "\n${YELLOW}📝 Creating convenience scripts...${NC}"

# Training script
cat > "$SCRIPT_DIR/run_training.sh" << 'EOF'
#!/bin/bash
# Quick training script for multimodal motion fusion

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Starting multimodal training..."
echo "📊 Monitor with: tensorboard --logdir logs"

# Default to taichi-boxing fusion experiment
EXPERIMENT=${1:-"exp/taichi_boxing_fusion"}

python train_multimodal.py --config-name="$EXPERIMENT"
EOF

chmod +x "$SCRIPT_DIR/run_training.sh"

# Evaluation script
cat > "$SCRIPT_DIR/run_evaluation.sh" << 'EOF'
#!/bin/bash
# Quick evaluation script for multimodal models

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "🔍 Starting multimodal evaluation..."

# Default model path
MODEL_PATH=${1:-"checkpoints/latest.pt"}

python eval_multimodal.py model_path="$MODEL_PATH"
EOF

chmod +x "$SCRIPT_DIR/run_evaluation.sh"

# Monitoring script
cat > "$SCRIPT_DIR/monitor_training.sh" << 'EOF'
#!/bin/bash
# Monitor training progress

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "📊 Starting monitoring tools..."

# Start tensorboard in background
tensorboard --logdir="$SCRIPT_DIR/logs" --port=6006 &
TENSORBOARD_PID=$!

echo "🌐 TensorBoard available at: http://localhost:6006"
echo "📊 Press Ctrl+C to stop monitoring"

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping monitoring..."
    kill $TENSORBOARD_PID 2>/dev/null
    exit 0
}

trap cleanup INT

# Keep script running
wait $TENSORBOARD_PID
EOF

chmod +x "$SCRIPT_DIR/monitor_training.sh"

print_status $? "Convenience scripts creation"

# Step 5: Validate installation
echo -e "\n${YELLOW}🔍 Validating installation...${NC}"
python3 integrate_multimodal.py --validate-only
print_status $? "Installation validation"

# Step 6: Create README for quick start
cat > "$SCRIPT_DIR/QUICKSTART.md" << 'EOF'
# Multimodal Motion Fusion - Quick Start

## 🚀 Training

### Basic Training (Taichi + Boxing)
```bash
./run_training.sh
```

### Full Multimodal Training (All 6 motion types)
```bash
./run_training.sh exp/full_multimodal
```

### Custom Configuration
```bash
python train_multimodal.py --config-name=your_config
```

## 📊 Monitoring

### Start TensorBoard
```bash
./monitor_training.sh
```

### View logs
```bash
ls -la logs/
```

## 🔍 Evaluation

### Evaluate Latest Model
```bash
./run_evaluation.sh
```

### Evaluate Specific Model
```bash
./run_evaluation.sh path/to/your/model.pt
```

## 📁 Directory Structure

```
innovation_multimodal/
├── config/                 # Configuration files
│   ├── multimodal_base.yaml
│   ├── env/
│   ├── algo/
│   └── exp/
├── motion_encoder.py       # Motion encoding module
├── fusion_controller.py    # Fusion control module
├── multimodal_env.py      # Environment extension
├── multimodal_ppo.py      # Algorithm implementation
├── train_multimodal.py    # Training script
├── eval_multimodal.py     # Evaluation script
├── logs/                  # Training logs
├── checkpoints/           # Model checkpoints
└── evaluation/            # Evaluation results
```

## 🎯 Experiment Configurations

- `exp/taichi_boxing_fusion.yaml` - Taichi + Boxing fusion
- `exp/full_multimodal.yaml` - All 6 motion types
- Create custom configs in `config/exp/`

## 🔧 Troubleshooting

### Common Issues
1. **Import errors**: Ensure PBHC is in Python path
2. **CUDA issues**: Check GPU availability and PyTorch installation
3. **Config errors**: Validate YAML syntax in config files

### Debug Mode
Add `hydra.verbose=true` to any training command for detailed logging.
EOF

echo -e "\n${GREEN}🎉 Setup complete!${NC}"
echo -e "\n${BLUE}📖 Quick start guide: $SCRIPT_DIR/QUICKSTART.md${NC}"
echo -e "\n${YELLOW}🚀 Ready to start training:${NC}"
echo -e "   cd $SCRIPT_DIR"
echo -e "   ./run_training.sh"
echo -e "\n${YELLOW}📊 Monitor progress:${NC}"
echo -e "   ./monitor_training.sh"
