; Inno Setup script for Revolvers (Orbitals PC Build-From-Dump Installer)
; Contains ZERO Switch 2 game code or assets; requires user's own .NUP or extracted exefs/romfs folder.

#define MyAppName "Orbitals"
#define MyAppVersion "1.0"
#define MyAppExeName "orbitals.exe"
#define StageDir "stage"

[Setup]
AppId={{B7E41A92-8D3C-4E15-9F02-4A91D3C8E701}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={sd}\Games\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
DirExistsWarning=no
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=Orbitals-Builder-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Stage the clean build kit, open-source shader translator sidecars, and orchestrator into {tmp}
; (automatically cleaned up when Setup exits)
Source: "{#StageDir}\*"; DestDir: "{tmp}\orbitals-builder"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File ""{tmp}\orbitals-builder\install-orbitals.ps1"" -KitDir ""{tmp}\orbitals-builder\kit"" -SidecarDir ""{tmp}\orbitals-builder\sidecars"" -IniFile ""{tmp}\orbitals-builder\nx2navi.ini"" -OutDir ""{app}"" -SourceType {code:GetSourceType} -SourcePath ""{code:GetSourcePath}"""; StatusMsg: "Building Orbitals from your Switch 2 copy (this takes ~2-5 minutes)..."; Flags: waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent; Check: BuiltExeExists

[UninstallDelete]
; Remove generated PC build & extracted assets while preserving user saves (profile\) and settings (nx2navi.ini)
Type: files; Name: "{app}\orbitals.exe"
Type: files; Name: "{app}\*.dll"
Type: files; Name: "{app}\*.exe"
Type: files; Name: "{app}\*.json"
Type: files; Name: "{app}\*.bin"
Type: files; Name: "{app}\nx2navi_xlate.id"
Type: files; Name: "{app}\nx2navi.log"
Type: files; Name: "{app}\build-provenance.json"
Type: filesandordirs; Name: "{app}\exefs"
Type: filesandordirs; Name: "{app}\romfs"
Type: filesandordirs; Name: "{app}\shader-cache"
Type: filesandordirs; Name: "{app}\build-cache"

[Code]
var
  SourceTypePage: TInputOptionWizardPage;
  NupPage: TInputFileWizardPage;
  FolderPage: TInputDirWizardPage;

procedure InitializeWizard;
begin
  SourceTypePage := CreateInputOptionPage(
    wpSelectDir,
    'Select Your Switch 2 Copy of Orbitals',
    'An original copy of Orbitals (Switch 2) is required to build and install the PC version.',
    'Choose the format of your dumped Switch 2 copy:',
    True,
    False
  );
  SourceTypePage.Add('Switch 2 .NUP file (Recommended)');
  SourceTypePage.Add('Extracted Switch 2 game folder (contains exefs/ and romfs/)');
  SourceTypePage.SelectedValueIndex := 0;

  NupPage := CreateInputFilePage(
    SourceTypePage.ID,
    'Locate Your Orbitals .NUP File',
    'Select your dumped Switch 2 .nup package for Orbitals.',
    'Setup will verify the game ID and build the PC version directly from this file.'
  );
  NupPage.Add('Orbitals .nup file:', 'Switch 2 NUP files (*.nup)|*.nup|All files (*.*)|*.*', '.nup');

  FolderPage := CreateInputDirPage(
    NupPage.ID,
    'Locate Your Extracted Orbitals Folder',
    'Select the folder containing the extracted exefs and romfs directories.',
    'The selected folder must contain both an "exefs" subfolder and a "romfs" subfolder.',
    False,
    ''
  );
  FolderPage.Add('Extracted game folder:');
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  if PageID = NupPage.ID then
    Result := (SourceTypePage.SelectedValueIndex <> 0)
  else if PageID = FolderPage.ID then
    Result := (SourceTypePage.SelectedValueIndex <> 1);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Path: String;
begin
  Result := True;
  if CurPageID = NupPage.ID then
  begin
    Path := Trim(NupPage.Values[0]);
    if (Path = '') or (not FileExists(Path)) then
    begin
      MsgBox('Please select a valid Orbitals .nup file to continue.', mbError, MB_OK);
      Result := False;
    end;
  end
  else if CurPageID = FolderPage.ID then
  begin
    Path := RemoveBackslashUnlessRoot(Trim(FolderPage.Values[0]));
    if (Path = '') or (not DirExists(Path + '\exefs')) or (not DirExists(Path + '\romfs')) then
    begin
      MsgBox('The selected folder must contain both "exefs" and "romfs" subdirectories.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

function GetSourceType(Param: String): String;
begin
  if SourceTypePage.SelectedValueIndex = 0 then
    Result := 'nup'
  else
    Result := 'folder';
end;

function GetSourcePath(Param: String): String;
begin
  if SourceTypePage.SelectedValueIndex = 0 then
    Result := Trim(NupPage.Values[0])
  else
    Result := RemoveBackslashUnlessRoot(Trim(FolderPage.Values[0]));
end;

function BuiltExeExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\{#MyAppExeName}'));
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  S: String;
begin
  S := '';
  if MemoDirInfo <> '' then
    S := S + MemoDirInfo + NewLine + NewLine;
  S := S + 'Switch 2 Game Source:' + NewLine;
  if SourceTypePage.SelectedValueIndex = 0 then
    S := S + Space + 'NUP file: ' + Trim(NupPage.Values[0]) + NewLine + NewLine
  else
    S := S + Space + 'Extracted folder: ' + RemoveBackslashUnlessRoot(Trim(FolderPage.Values[0])) + NewLine + NewLine;
  if MemoGroupInfo <> '' then
    S := S + MemoGroupInfo + NewLine + NewLine;
  if MemoTasksInfo <> '' then
    S := S + MemoTasksInfo + NewLine;
  Result := S;
end;
