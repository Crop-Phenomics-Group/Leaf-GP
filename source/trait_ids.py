

# class that stores mapping between csv column ids as enums and text-friendly descriptions
class Trait:

    # enums for csv column names
    IMAGE_NAME = "Img_name"
    EXP_DATE = "EXP_Date"
    EXPERIMENT_REF = "Genotype_Treatment"
    TRAY_NO = "Tray_No"
    POT_ID = "Pot_ID"
    POT_X = "Pot_X"
    POT_Y = "Pot_Y"
    PROJECTED_LEAF_AREA = "Projected_LeafArea(mm^2)"
    LEAF_PERIMETER = "Leaf_perimeter(mm)"
    CANOPY_LENGTH = "Canopy_Length(mm)"
    CANOPY_WIDTH = "Canopy_Width(mm)"
    STOCKINESS = "Stockiness(%)"
    LEAF_STOCKINESS = "Leaf_Stockiness"
    LEAF_CANOPY_SIZE = "Leaf_CanopySize(mm^2)"
    LEAF_COMPACTNESS = "Leaf_Compactness(%)"
    LARGE_LEAF_NO = "Large_Leaf_No"
    LEAF_TOTAL_NO = "Leaf_TotalNo"
    GREENNESS = "Greenness(0-255)"
    PIX2MM_RATIO = "pix2mm2_ratio"

    # mapping between column ids and text-friendly descriptions (e.g. for GUI table columns)
    TRAIT_IDS = {   PROJECTED_LEAF_AREA: "Projected Leaf Area (mm^2)",
                    LEAF_PERIMETER: "Leaf Perimeter (mm)",
                    CANOPY_LENGTH: "Canopy Length (mm)",
                    STOCKINESS: "Stockiness (%)",
                    LEAF_STOCKINESS: "Leaf Stockiness",
                    LEAF_CANOPY_SIZE: "Leaf Canopy Size (mm^2)",
                    LEAF_COMPACTNESS: "Compactness (%)",
                    LEAF_TOTAL_NO: "Total No. Leaves"
    }

    # list of column id enums for attributes that will be plotted when analysing arabidopsis datasets
    ARABIDOPSIS_PLOT_IDS = [PROJECTED_LEAF_AREA, LEAF_PERIMETER, CANOPY_LENGTH, STOCKINESS, LEAF_CANOPY_SIZE, LEAF_COMPACTNESS, LEAF_TOTAL_NO]
    # list of column id enums for attributes that will be plotted when analysing wheat datasets
    WHEAT_PLOT_IDS = [PROJECTED_LEAF_AREA, CANOPY_LENGTH, LEAF_STOCKINESS, LEAF_COMPACTNESS]
    # list of column id enums for attributes that will be plotted when no experimental data available
    NO_EXPERIMENTAL_DATA_PLOT_IDS = []


