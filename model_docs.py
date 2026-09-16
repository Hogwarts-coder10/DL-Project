import os
from src.models.unetplusplus import build_unet_plus_plus

def generate_docs():
    # 1. Build the model with your exact dataset specs
    model = build_unet_plus_plus(input_shape=(256, 256, 3), num_classes=4)
    
    # 2. Print the standard summary to the terminal
    print("\n" + "="*50)
    print(" U-NET++ ARCHITECTURE SPECS")
    print("="*50)
    model.summary()
    
    # 3. Save the exact layer map and param count to a text file for your documentation
    os.makedirs("docs", exist_ok=True)
    with open('docs/unetplusplus_summary.txt', 'w') as f:
        # Pass a custom print function to write the summary directly to the file
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    print("\n--> Documentation saved successfully to docs/unetplusplus_summary.txt")

if __name__ == "__main__":
    generate_docs()
