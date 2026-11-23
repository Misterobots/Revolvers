// The settings menu's palette, taken from Orbitals' own title screen: the deep-space navy of its backgrounds for the
// panels, the blue of its START button for what is selected or focused, the peach of the logo for titles and the
// red of the star for warnings. One place, so every page draws in the same colours.
#pragma once
#include "imgui.h"

// as ImVec4 (style colours, TextColored)
static const ImVec4 kThBlue(0.38f, 0.66f, 0.96f, 1);        // START button: focus, check marks, slider grabs
static const ImVec4 kThBlueDeep(0.16f, 0.36f, 0.62f, 1);    // selected tab, active control
static const ImVec4 kThBlueDim(0.11f, 0.22f, 0.38f, 1);     // hovered control
static const ImVec4 kThPanel(0.035f, 0.060f, 0.115f, 0.95f); // window background
static const ImVec4 kThFrame(0.07f, 0.12f, 0.21f, 1);       // fields, buttons, tabs at rest
static const ImVec4 kThPeach(0.96f, 0.84f, 0.75f, 1);       // logo: titles and headings
static const ImVec4 kThText(0.91f, 0.94f, 0.98f, 1);
static const ImVec4 kThTextDim(0.55f, 0.62f, 0.72f, 1);
static const ImVec4 kThRed(0.95f, 0.36f, 0.30f, 1);         // the red star: errors
static const ImVec4 kThWarm(0.98f, 0.62f, 0.50f, 1);        // warnings and notes

// as packed colours (draw lists)
#define TH_BLUE(a) IM_COL32(96, 170, 245, a)
#define TH_PEACH(a) IM_COL32(246, 214, 190, a)
#define TH_TEXT(a) IM_COL32(232, 240, 250, a)
#define TH_DIM IM_COL32(140, 156, 178, 255)
#define TH_GREY IM_COL32(92, 104, 122, 255)
#define TH_NOTE IM_COL32(246, 160, 130, 255)
#define TH_NOTE_OFF IM_COL32(190, 130, 112, 255)
#define TH_SHADE IM_COL32(4, 8, 18, 150)   // behind the menu window
