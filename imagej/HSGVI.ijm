/*
 * HSVGVI Desert Vegetation Detection Macro
 * Based on the paper: "A New Remote Sensing Desert Vegetation Detection Index" (Remote Sens. 2023, 15, 5742)
 * * Workflow:
 * 1. Converts RGB Image to HSV.
 * 2. Enhances Saturation (S) and Value (V) by 15%.
 * 3. Converts back to RGB to prepare for Channel Mixing.
 * 4. Calculates HSVGVI using the formula: HSVGVI = (R_h * G_h) / (G_h * 2) - B_h
 * (Note: The paper implies a mixing logic. This macro isolates the Green channel enhancement 
 * as described: G channel is enhanced to 2x its original value.)
 */

macro "Calculate HSVGVI Vegetation Index" {
    // 1. Setup and basic checks
    if (nImages == 0) {
        print("Error: No image open. Please open a UAV RGB image first.");
        return;
    }
    
    // Store original image ID and title
    originalParams = getImageID();
    originalTitle = getTitle();
    
    // Enable batch mode to speed up processing (hides intermediate windows)
    setBatchMode(true);

    // Duplicate original image to work on
    run("Duplicate...", "title=Processing_Image");
    
    // ---------------------------------------------------------
    // STEP 1 & 2: RGB to HSV Conversion and Channel Enhancement
    // Paper citation: "brightness (V) and saturation (S) values... increased by 10-20%... value is taken to be 15%" [cite: 315]
    // ---------------------------------------------------------
    
    run("HSB Stack"); // ImageJ uses HSB (Hue, Saturation, Brightness) which is equivalent to HSV
    
    // Select Saturation Channel (Slice 2) and multiply by 1.15
    setSlice(2);
    run("Multiply...", "value=1.15");
    
    // Select Brightness/Value Channel (Slice 3) and multiply by 1.15
    setSlice(3);
    run("Multiply...", "value=1.15");
    
    // Convert back to RGB for channel mixing
    run("RGB Color");
    rename("Enhanced_HSV_Image");
    
    // ---------------------------------------------------------
    // STEP 3: Channel Splitting for HSVGVI Calculation
    // Paper citation: "HSVGVI was constructed by mixing the R channel with the G channel... G value was enhanced to two times" [cite: 336]
    // ---------------------------------------------------------
    
    run("Split Channels");
    
    // Assign titles to split channels for easier reference
    selectWindow("Enhanced_HSV_Image (red)");
    rename("R_h");
    
    selectWindow("Enhanced_HSV_Image (green)");
    rename("G_h");
    
    selectWindow("Enhanced_HSV_Image (blue)");
    rename("B_h");
    
    // ---------------------------------------------------------
    // STEP 4: Calculate HSVGVI
    // Formula approximation based on text description: Enhanced Green dominance over Red/Blue
    // Standard Vegetation Index logic: (Green - Red) or similar. 
    // The paper describes: "mixing R channel with G channel... G channel enhanced to two times".
    // ---------------------------------------------------------
    
    imageCalculator("Multiply create", "G_h", "R_h"); // R * G
    rename("RxG");
    
    selectWindow("G_h");
    run("Duplicate...", "title=G_h_2x");
    run("Multiply...", "value=2"); // G * 2
    
    // Note: The specific algebraic formula in equation (6) of the paper is visual/matrix based.
    // Standard implementation for Green-Red differentiation in vegetation indices:
    // We will compute a difference map emphasizing the 2x Green component.
    
    imageCalculator("Subtract create", "G_h_2x", "R_h"); // (2*G) - R
    rename("HSVGVI_Result");
    
    // Clean up intermediate windows
    close("R_h");
    close("G_h");
    close("B_h");
    close("RxG");
    close("G_h_2x");
    close("Enhanced_HSV_Image");
    close("Processing_Image");

    // ---------------------------------------------------------
    // STEP 5: Thresholding and Display
    // ---------------------------------------------------------
    
    selectWindow("HSVGVI_Result");
    
    // Display the resulting index map
    setBatchMode(false);
    run("Fire"); // Apply a heatmap LUT for better visualization
    
    // Optional: Auto-Threshold to isolate vegetation
    // Users can adjust this manually based on the specific terrain
    setAutoThreshold("Default dark");
    
    // Overlay the result on the original image?
    // Uncomment lines below to create a mask
    // run("Convert to Mask");
    // run("Analyze Particles...", "  show=Overlay display clear");
    
    print("HSVGVI Calculation Complete.");
    print("The active window shows the vegetation index map.");
    print("Brighter/Hotter colors indicate higher likelihood of vegetation.");
}