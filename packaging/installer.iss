; Inno Setup script for AI Subtitle Studio. Compiled by packaging\build_windows.ps1:
;   ISCC.exe /DMyAppVersion=1.0.0 packaging\installer.iss
; Per-user install (no administrator rights needed); user data in %LOCALAPPDATA%\AISubtitleStudio is kept on
; uninstall. NVIDIA GPU libraries are downloaded by the app on first use (D-043).

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#define MyAppName "AI Subtitle Studio"
#define MyAppExe "AISubtitleStudio.exe"
#define SourceDir "..\dist\AI Subtitle Studio"

[Setup]
AppId={{6B0D9E3A-4C55-4E7B-9C1E-3F2B8A41D7C2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=AI Subtitle Studio
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=AI-Subtitle-Studio-Setup
SetupIconFile=..\app\resources\icons\app.ico
UninstallDisplayIcon={app}\{#MyAppExe}
; The Inno compiler is a 32-bit program: ultra64 (64 MB dictionary) with 4 threads ran out of memory on a payload
; of several hundred MB ("Error ... Out of memory" on the GitHub runner). max + a separate process + 2 threads fits.
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=2
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
#if FileExists(AddBackslash(CompilerPath) + "Languages\Arabic.isl")
Name: "arabic"; MessagesFile: "compiler:Languages\Arabic.isl"
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Files of an older version must not stay next to the new ones (removed modules, renamed DLLs).
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; IconFilename: "{app}\{#MyAppExe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; IconFilename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
; An in-app update runs the installer silently: start the program again afterwards.
Filename: "{app}\{#MyAppExe}"; Flags: nowait; Check: WizardSilent
