Unicode true

!include "MUI2.nsh"
!include "LogicLib.nsh"

!ifndef SEVENZIP_EXE
    !error "SEVENZIP_EXE is required"
!endif
!ifndef OUTPUT_FILE
    !error "OUTPUT_FILE is required"
!endif
!ifndef ARCHIVE_BASE
    !error "ARCHIVE_BASE is required"
!endif
!ifndef ARCHIVE_VOLUME_COUNT
    !error "ARCHIVE_VOLUME_COUNT is required"
!endif
!ifndef APP_VERSION
    !error "APP_VERSION is required"
!endif
!ifndef FILE_VERSION
    !error "FILE_VERSION is required"
!endif
!ifndef ICON_FILE
    !error "ICON_FILE is required"
!endif

Name "一点筛图"
OutFile "${OUTPUT_FILE}"
InstallDir "$LOCALAPPDATA\Programs\一点筛图"
InstallDirRegKey HKCU "Software\一点筛图" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetOverwrite on
ShowInstDetails show
Icon "${ICON_FILE}"
UninstallIcon "${ICON_FILE}"

VIProductVersion "${FILE_VERSION}"
VIAddVersionKey /LANG=2052 "ProductName" "一点筛图"
VIAddVersionKey /LANG=2052 "FileDescription" "一点筛图 CUDA 本地安装器"
VIAddVersionKey /LANG=2052 "FileVersion" "${APP_VERSION}"
VIAddVersionKey /LANG=2052 "LegalCopyright" "Copyright © 2026 一点筛图"

!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\一点筛图.exe"
!define MUI_FINISHPAGE_RUN_TEXT "运行一点筛图"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"

Var PayloadIndex
Var PayloadSuffix

Function VerifyPayload
    StrCpy $PayloadIndex 1

payload_loop:
    IntCmp $PayloadIndex ${ARCHIVE_VOLUME_COUNT} payload_check payload_check payload_ready

payload_check:
    StrCpy $PayloadSuffix "00$PayloadIndex"
    StrCpy $PayloadSuffix $PayloadSuffix 3 -3
    IfFileExists "$EXEDIR\${ARCHIVE_BASE}.$PayloadSuffix" payload_next 0
    MessageBox MB_ICONSTOP|MB_OK "缺少 ${ARCHIVE_BASE}.$PayloadSuffix。请把安装器与全部 ${ARCHIVE_VOLUME_COUNT} 个分卷放在同一目录。"
    Abort

payload_next:
    IntOp $PayloadIndex $PayloadIndex + 1
    Goto payload_loop

payload_ready:
FunctionEnd

Section "一点筛图 CUDA" SEC_MAIN
    Call VerifyPayload
    SetOutPath "$PLUGINSDIR"
    File /oname=7za.exe "${SEVENZIP_EXE}"

    DetailPrint "正在校验并解压 CUDA 程序与本地 AI 模型……"
    ExecWait '"$PLUGINSDIR\7za.exe" x -y -aoa "-o$INSTDIR" "$EXEDIR\${ARCHIVE_BASE}.001"' $0
    ${If} $0 != 0
        MessageBox MB_ICONSTOP|MB_OK "资源解压失败，7-Zip 返回代码 $0。请核对 SHA256 后重试。"
        Abort
    ${EndIf}

    IfFileExists "$INSTDIR\一点筛图.exe" app_ready 0
    MessageBox MB_ICONSTOP|MB_OK "安装内容不完整：没有找到一点筛图.exe。"
    Abort

app_ready:
    WriteUninstaller "$INSTDIR\卸载一点筛图.exe"
    WriteRegStr HKCU "Software\一点筛图" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\一点筛图" "DisplayName" "一点筛图"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\一点筛图" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\一点筛图" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\一点筛图" "UninstallString" '"$INSTDIR\卸载一点筛图.exe"'
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\一点筛图" "EstimatedSize" 6156100

    CreateDirectory "$SMPROGRAMS\一点筛图"
    CreateShortcut "$SMPROGRAMS\一点筛图\一点筛图.lnk" "$INSTDIR\一点筛图.exe"
    CreateShortcut "$DESKTOP\一点筛图.lnk" "$INSTDIR\一点筛图.exe"
SectionEnd

Section "Uninstall"
    Delete "$DESKTOP\一点筛图.lnk"
    Delete "$SMPROGRAMS\一点筛图\一点筛图.lnk"
    RMDir "$SMPROGRAMS\一点筛图"
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\一点筛图"
    DeleteRegKey HKCU "Software\一点筛图"
    RMDir /r "$INSTDIR"
SectionEnd
