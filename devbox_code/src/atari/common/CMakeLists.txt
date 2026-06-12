target_sources(Algorithm
        PRIVATE
        ColourPalette.cpp
        Constants.cpp
        Log.cpp
        Palettes.hpp
        ScreenExporter.cpp
        SoundExporter.cpp
        SoundNull.cxx
        SoundSDL.cxx
        SDL2.cpp
        DynamicLoad.cpp
        ScreenSDL.cpp
)

#
#target_sources(Control
#        PRIVATE
#        ColourPalette.cpp
#        Constants.cpp
#        Log.cpp
#        Palettes.hpp
#        ScreenExporter.cpp
#        SoundExporter.cpp
#        SoundNull.cxx
#        SoundSDL.cxx
#        SDL2.cpp
#        DynamicLoad.cpp
#        ScreenSDL.cpp
#)


# access global list called "global_list" made in the parent directory and append the following files
# to it. This is done so that the files are only compiled once and can be
# used in multiple targets



#target_sources(Torch
#        PRIVATE ColourPalette.cpp
#        Constants.cpp
#        Log.cpp
#        Palettes.hpp
#        ScreenExporter.cpp
#        SoundExporter.cpp
#        SoundNull.cxx
#        SoundSDL.cxx
#        SDL2.cpp
#        DynamicLoad.cpp
#        ScreenSDL.cpp)
