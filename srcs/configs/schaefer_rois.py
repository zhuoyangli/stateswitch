"""
Schaefer 400 (17-Network) ROI definitions.

All parcel IDs are 1-based, matching the Schaefer atlas label ordering.
Each ROI is a dict with 'left' and/or 'right' parcel ID lists.
"""

# ============================================================================
# Primary ROIs
# ============================================================================

EARLY_VISUAL = {
    'name': 'Early Visual Cortex',
    'abbreviation': 'EVC',
    'left': [7, 18, 19, 20],
    'left_labels': [
        '17Networks_LH_VisCent_Striate_1',
        '17Networks_LH_VisPeri_StriCal_1',
        '17Networks_LH_VisPeri_StriCal_2',
        '17Networks_LH_VisPeri_ExStrSup_1',
    ],
    'right': [207, 218, 219],
    'right_labels': [
        '17Networks_RH_VisCent_Striate_1',
        '17Networks_RH_VisPeri_StriCal_1',
        '17Networks_RH_VisPeri_StriCal_2',
    ],
    'sources': [
        'Lee & Chen, 2022 (Nature Communications)',
        'Lee et al., 2025 (preprint)',
    ],
}

EARLY_AUDITORY = {
    'name': 'Early Auditory Cortex',
    'abbreviation': 'EAC',
    'left': [44, 45, 46],
    'left_labels': [
        '17Networks_LH_SomMotB_Aud_1',
        '17Networks_LH_SomMotB_Aud_2',
        '17Networks_LH_SomMotB_Ins_1',
    ],
    'right': [244, 245, 246],
    'right_labels': [
        '17Networks_RH_SomMotB_Aud_1',
        '17Networks_RH_SomMotB_Aud_2',
        '17Networks_RH_SomMotB_Ins_1',
    ],
    'sources': [
        'Lee et al., 2025 (preprint)',
        'Zuo et al., 2020 (NeuroImage)',
    ],
}

POSTERIOR_MEDIAL = {
    'name': 'Posterior Medial Cortex',
    'abbreviation': 'PMC',
    'left': [154, 155, 156, 157, 158, 159, 160],
    'left_labels': [
        f'17Networks_LH_DefaultA_pCunPCC_{i}' for i in range(1, 8)
    ],
    'right': [363, 364, 365, 366, 367],
    'right_labels': [
        f'17Networks_RH_DefaultA_pCunPCC_{i}' for i in range(1, 6)
    ],
    'sources': [
        'Lee & Chen, 2022 (Nature Communications)',
        'Lee et al., 2025 (preprint)',
        'Zuo et al., 2020 (NeuroImage)',
    ],
}

ANGULAR_GYRUS = {
    'name': 'Angular Gyrus',
    'abbreviation': 'AG',
    'left': [149, 150, 173, 174, 188],
    'left_labels': [
        '17Networks_LH_DefaultA_IPL_1',
        '17Networks_LH_DefaultA_IPL_2',
        '17Networks_LH_DefaultB_IPL_1',
        '17Networks_LH_DefaultB_IPL_2',
        '17Networks_LH_DefaultC_IPL_1',
    ],
    'right': [359, 360, 385, 386],
    'right_labels': [
        '17Networks_RH_DefaultA_IPL_1',
        '17Networks_RH_DefaultA_IPL_2',
        '17Networks_RH_DefaultC_IPL_1',
        '17Networks_RH_DefaultC_IPL_2',
    ],
    'sources': [
        'Lee et al., 2025 (preprint)',
    ],
}

RIGHT_TPJ = {
    'name': 'Right Temporoparietal Junction',
    'abbreviation': 'rTPJ',
    'left': [],
    'right': [396, 397, 398, 399, 400],
    'right_labels': [
        '17Networks_RH_TempPar_6',
        '17Networks_RH_TempPar_7',
        '17Networks_RH_TempPar_8',
        '17Networks_RH_TempPar_9',
        '17Networks_RH_TempPar_10',
    ],
}

# ============================================================================
# Additional ROIs (Zuo et al., 2020)
# ============================================================================

LEFT_STG = {
    'name': 'Left Superior Temporal Gyrus',
    'abbreviation': 'L-STG',
    'left': [44, 45, 46, 196, 197],
    'right': [],
    'sources': ['Zuo et al., 2020 (NeuroImage)'],
}

LEFT_PLP = {
    'name': 'Left Posterior Lateral Parietal Cortex',
    'abbreviation': 'L-PLP',
    'left': [149, 150, 173, 174, 188, 136, 137, 138],
    'right': [],
    'sources': ['Zuo et al., 2020 (NeuroImage)'],
}

RIGHT_PLP = {
    'name': 'Right Posterior Lateral Parietal Cortex',
    'abbreviation': 'R-PLP',
    'left': [],
    'right': [359, 360, 385, 386, 338, 339, 340, 341, 304],
    'sources': ['Zuo et al., 2020 (NeuroImage)'],
}

DORSAL_MPFC = {
    'name': 'Left and Dorsal Medial Prefrontal Cortex',
    'abbreviation': 'dmPFC',
    'left': [161, 162, 163, 164, 165, 166, 175, 176, 177, 178, 179, 180],
    'right': [],
    'sources': ['Zuo et al., 2020 (NeuroImage)'],
}

RIGHT_MPFC = {
    'name': 'Right Medial Prefrontal Cortex',
    'abbreviation': 'R-mPFC',
    'left': [],
    'right': [368, 369, 370, 371, 372, 373],
    'sources': ['Zuo et al., 2020 (NeuroImage)'],
}

MPFC = {
    'name': 'Medial Prefrontal Cortex',
    'abbreviation': 'mPFC',
    'left': [161, 162, 163, 164, 165, 166],
    'left_labels': [f'17Networks_LH_DefaultA_PFCm_{i}' for i in range(1, 7)],
    'right': [368, 369, 370, 371, 372],
    'right_labels': [f'17Networks_RH_DefaultA_PFCm_{i}' for i in range(1, 6)],
}

RIGHT_PHC_TP = {
    'name': 'Right Parahippocampal Cortex and Temporal Pole',
    'abbreviation': 'R-PHC/TP',
    'left': [],
    'right': [389, 319, 320, 322, 376],
    'sources': ['Zuo et al., 2020 (NeuroImage)'],
}

DLPFC = {
    'name': 'Dorsolateral Prefrontal Cortex',
    'abbreviation': 'dlPFC',
    # ContA_PFCl_2/3 on LH + ContB_PFCld on RH
    # (removed LH ContA_PFCl_1 and all RH ContA_PFCl parcels)
    # Note: ContB_PFCld is RH-only in the Schaefer 400 17-network atlas
    'left': [131, 132],
    'left_labels': [
        '17Networks_LH_ContA_PFCl_2',
        '17Networks_LH_ContA_PFCl_3',
    ],
    'right': [342, 343, 344, 345],
    'right_labels': [
        '17Networks_RH_ContB_PFCld_1',
        '17Networks_RH_ContB_PFCld_2',
        '17Networks_RH_ContB_PFCld_3',
        '17Networks_RH_ContB_PFCld_4',
    ],
}

VLPFC = {
    'name': 'Ventrolateral Prefrontal Cortex',
    'abbreviation': 'vlPFC',
    # ContA_PFClv + ContB_PFClv parcels (bilateral)
    'left': [129, 130, 141, 142, 143],
    'left_labels': [
        '17Networks_LH_ContA_PFClv_1',
        '17Networks_LH_ContA_PFClv_2',
        '17Networks_LH_ContB_PFClv_1',
        '17Networks_LH_ContB_PFClv_2',
        '17Networks_LH_ContB_PFClv_3',
    ],
    'right': [347, 348, 349, 350],
    'right_labels': [
        '17Networks_RH_ContB_PFClv_1',
        '17Networks_RH_ContB_PFClv_2',
        '17Networks_RH_ContB_PFClv_3',
        '17Networks_RH_ContB_PFClv_4',
    ],
}

DACC = {
    'name': 'Dorsal Anterior Cingulate Cortex',
    'abbreviation': 'dACC',
    # Derived by mapping 7-network SalVentAttn_Med_1/2/4 parcels to 17-network
    # via vertex-level overlap on fsaverage6 surface (annot files).
    # LH: SalVentAttnB_PFCmp_1, SalVentAttnA_FrMed_1, SalVentAttnA_FrMed_2
    # RH: SalVentAttnB_PFCmp_2, SalVentAttnA_FrMed_1
    'left': [108, 98, 99],
    'left_labels': [
        '17Networks_LH_SalVentAttnB_PFCmp_1',
        '17Networks_LH_SalVentAttnA_FrMed_1',
        '17Networks_LH_SalVentAttnA_FrMed_2',
    ],
    'right': [312, 296],
    'right_labels': [
        '17Networks_RH_SalVentAttnB_PFCmp_2',
        '17Networks_RH_SalVentAttnA_FrMed_1',
    ],
}

# ============================================================================
# Convenience groupings
# ============================================================================

PRIMARY_ROIS = {
    'evc': EARLY_VISUAL,
    'eac': EARLY_AUDITORY,
    'pmc': POSTERIOR_MEDIAL,
    'ag': ANGULAR_GYRUS,
    'rtpj': RIGHT_TPJ,
}

ZUO_ROIS = {
    'l_stg': LEFT_STG,
    'l_plp': LEFT_PLP,
    'r_plp': RIGHT_PLP,
    'dmpfc': DORSAL_MPFC,
    'r_mpfc': RIGHT_MPFC,
    'r_phc_tp': RIGHT_PHC_TP,
    'mpfc': MPFC,
}

DACC_ROIS = {
    'dlpfc': DLPFC,
    'dacc': DACC,
    'vlpfc': VLPFC,
}

ALL_ROIS = {**PRIMARY_ROIS, **ZUO_ROIS, **DACC_ROIS}


def get_bilateral_ids(roi):
    """Return combined left + right parcel IDs for an ROI."""
    return roi.get('left', []) + roi.get('right', [])
