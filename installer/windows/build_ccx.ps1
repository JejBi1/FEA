# Builds the Windows CalculiX of FEA Studio (ccx_spooles.exe) from pinned sources and puts it, with a build manifest, into
#   installer\windows\prebuilt\ccx_2.23_windows_x86_64.zip      (gitignored; installer\make_package.py reads it, installer\fea_setup.py unpacks it)
#
#   powershell -ExecutionPolicy Bypass -File installer\windows\build_ccx.ps1 [-Work tools\winccx_build] [-Archive PATH] [-Downloads DIR] [-Offline]
#                                                                           [-LibQuadmathSource DIR | -LibQuadmath DIR] [-SkipRuntimeSources] [-Jobs 4]
#
# No administrator rights and nothing is installed: the MSYS2 base system (pinned tarball, SHA-256 in installer\windows\ccx_toolchain.lock) is unpacked
# into the work folder, the pinned MSYS2 packages of the lock (gcc, gfortran, binutils, make, perl ...) are unpacked over it by build_ccx.sh, and
# the sources (CalculiX, SPOOLES, ARPACK-ng, LAPACK, the source packages of the run-time libraries) are taken with their SHA-256. The work folder is mapped to
# a drive letter (subst, default W:) so that the build runs on a short path without a space or a user name - none of them can get into the binary.
#
# -Downloads DIR  the local archive of everything that is downloaded (default <Work>\archive = tools\winccx_build\archive; gitignored): the MSYS2 base, the
#                 packages of the toolchain, the sources. A file of it whose SHA-256 matches the pin is used without the network; what is downloaded is
#                 stored there. installer\windows\fetch_ccx_archive.py fills and checks it without building.
# -Offline        never download (a file that is not in the archive stops the build) - the way to prove that the archive is complete.
# -LibQuadmathSource DIR  the unpacked source tree of GCC 16.2.0 (gcc-16.2.0\, from the source package in licenses\sources of the packages) with a modified
#                 libquadmath\: it is built with the toolchain of the build and the program is linked with it instead of the library of the toolchain
#                 (LGPL-2.1 section 6: relinking with a modified library; licenses\SOURCES.md describes it). A path without spaces. The manifest records it.
# -LibQuadmath DIR  the same with a ready libquadmath.a in DIR.
# -SkipRuntimeSources  do not fetch the MSYS2 source packages of the run-time libraries (the release package ships them; they are not needed to build).
# Takes 15-25 minutes with the network (about 430 MB downloaded the first time, the archive keeps it), 10-20 minutes from the archive. Needs Windows 10+
# (tar.exe reads the .tar.xz of the MSYS2 base).
[CmdletBinding()]
param(
    [string]$Work = "",
    [string]$Archive = "",
    [string]$Downloads = "",
    [switch]$Offline,
    [string]$LibQuadmath = "",
    [string]$LibQuadmathSource = "",
    [switch]$SkipRuntimeSources,
    [int]$Jobs = 4,
    [string]$Drive = "W"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent (Split-Path -Parent $here)
if (-not $Work) { $Work = Join-Path $root "tools\winccx_build" }
if (-not $Archive) { $Archive = Join-Path $here "prebuilt\ccx_2.23_windows_x86_64.zip" }
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$Work = (Resolve-Path $Work).Path
if (-not $Downloads) { $Downloads = Join-Path $Work "archive" }
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null
$Downloads = (Resolve-Path $Downloads).Path
if ($LibQuadmath) { $LibQuadmath = (Resolve-Path $LibQuadmath).Path }
if ($LibQuadmathSource) { $LibQuadmathSource = (Resolve-Path $LibQuadmathSource).Path }
# one build per work folder: a second run would delete the sources under the first one
$busy = [System.IO.File]::Open((Join-Path $Work ".build.lock"), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$lock = Join-Path $here "ccx_toolchain.lock"
$baseRow = Get-Content -Encoding UTF8 $lock | Where-Object { $_ -match '^base ' } | Select-Object -First 1
$null, $baseSum, $baseUrl = $baseRow -split ' '
$baseFile = Split-Path -Leaf $baseUrl
$drv = "$($Drive.TrimEnd(':').ToUpper()):"

# ---- the work folder on a drive letter
$mapped = $false
$cur = (& subst) | Where-Object { $_ -match "^$([regex]::Escape($drv))\\: => (.*)$" } | ForEach-Object { $Matches[1] }
if ($cur) {
    if ($cur.TrimEnd('\') -ne $Work.TrimEnd('\')) { throw "drive $drv is already mapped to $cur - use -Drive with a free letter" }
} elseif (Test-Path "$drv\") {
    throw "drive $drv exists - use -Drive with a free letter"
} else {
    & subst $drv $Work; if ($LASTEXITCODE) { throw "subst $drv failed" }
    $mapped = $true
}
try {
    # ---- the MSYS2 base: from the archive when its digest matches, otherwise downloaded into it
    $tarball = Join-Path $Downloads $baseFile
    if (-not (Test-Path $tarball) -or (Get-FileHash -Algorithm SHA256 $tarball).Hash.ToLower() -ne $baseSum) {
        if ($Offline) { throw "$baseFile is not in the archive $Downloads with the pinned SHA-256 and -Offline is set" }
        Write-Host "downloading $baseUrl"
        Invoke-WebRequest -Uri $baseUrl -OutFile "$tarball.part" -UseBasicParsing
        if ((Get-FileHash -Algorithm SHA256 "$tarball.part").Hash.ToLower() -ne $baseSum) { Remove-Item "$tarball.part"; throw "SHA-256 of $baseFile does not match the lock" }
        Move-Item -Force "$tarball.part" $tarball
    }
    $stamp = "$drv\msys64\.fea_base"
    if (-not (Test-Path $stamp) -or (Get-Content $stamp -Raw).Trim() -ne $baseSum) {
        if (Test-Path "$drv\msys64") { Write-Host "removing the old toolchain"; Remove-Item -Recurse -Force "$drv\msys64" }
        Write-Host "unpacking $baseFile (2 minutes: many small files)"
        & "$env:SystemRoot\System32\tar.exe" -xf $tarball -C "$drv\"
        if ($LASTEXITCODE) { throw "tar failed" }
        Set-Content -Path $stamp -Value $baseSum -NoNewline
    }

    # ---- the build inside the MSYS2 shell with a clean environment (nothing of the user's PATH)
    $bash = "$drv\msys64\usr\bin\bash.exe"
    $envExe = "$drv\msys64\usr\bin\env.exe"
    $cyg = "$drv\msys64\usr\bin\cygpath.exe"
    $scriptMsys = (& $cyg -u (Join-Path $here "build_ccx.sh"))
    # the archive inside the work folder is seen through the drive letter (no user name in any path of the build)
    $wk = $Work.TrimEnd('\') + '\'
    $dlSub = if ($Downloads.StartsWith($wk, [StringComparison]::OrdinalIgnoreCase)) { "$drv\" + $Downloads.Substring($wk.Length) } else { $Downloads }
    $archiveMsys = (& $cyg -u $dlSub)
    Remove-Item -Recurse -Force "$drv\out" -ErrorAction SilentlyContinue
    $dp = $drv.Substring(0, 1).ToLower()
    # strings of THIS machine that must not be in the binary (the script greps for them): the real work folder, the checkout, the archive, the profile, the user name
    $private = @($Work, $Work.Replace('\', '/'), $root, $root.Replace('\', '/'), $Downloads, $Downloads.Replace('\', '/'), $env:USERPROFILE, $env:USERPROFILE.Replace('\', '/'))
    if ($env:USERNAME -and $env:USERNAME.Length -ge 5) { $private += $env:USERNAME }
    $audit = ($private | Where-Object { $_ } | Select-Object -Unique) -join "|"
    $envArgs = @("PATH=/usr/bin", "MSYSTEM=UCRT64", "FEA_BUILD_JOBS=$Jobs", "FEA_AUDIT_STRINGS=$audit", "FEA_ARCHIVE=$archiveMsys", "HOME=/tmp", "TEMP=/tmp", "TMP=/tmp")
    if ($Offline) { $envArgs += "FEA_OFFLINE=1" }
    if ($LibQuadmath) { $envArgs += "FEA_LIBQUADMATH_DIR=$((& $cyg -u $LibQuadmath))" }
    if ($LibQuadmathSource) { $envArgs += "FEA_LIBQUADMATH_SOURCE=$((& $cyg -u $LibQuadmathSource))" }
    if ($SkipRuntimeSources) { $envArgs += "FEA_SKIP_RUNTIME_SOURCES=1" }
    & $envExe -i @envArgs $bash "$scriptMsys" "/$dp/out" "/$dp/work"
    if ($LASTEXITCODE) { throw "build_ccx.sh failed with code $LASTEXITCODE" }

    # ---- a smoke test of the binary that was made (it must start and say its version)
    $ver = (& "$drv\out\ccx_spooles.exe" -v 2>&1 | Out-String)
    if ($ver -notmatch "Version 2\.23") { throw "ccx_spooles.exe does not start or reports another version: $ver" }

    # ---- the archive: fixed order, fixed time stamps (a repeatable zip)
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Archive) | Out-Null
    Remove-Item -Force $Archive -ErrorAction SilentlyContinue
    $outDir = "$drv\out"
    $fixed = [DateTimeOffset]::new(2026, 11, 18, 0, 0, 0, [TimeSpan]::Zero)
    $zip = [System.IO.Compression.ZipFile]::Open($Archive, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $files = Get-ChildItem -Recurse -File $outDir | Sort-Object { $_.FullName.Substring($outDir.Length + 1).Replace('\', '/') } -CaseSensitive
        foreach ($f in $files) {
            $name = $f.FullName.Substring($outDir.Length + 1).Replace('\', '/')
            $e = $zip.CreateEntry($name, [System.IO.Compression.CompressionLevel]::Optimal)
            $e.LastWriteTime = $fixed
            $s = $e.Open(); $in = [System.IO.File]::OpenRead($f.FullName)
            try { $in.CopyTo($s) } finally { $in.Dispose(); $s.Dispose() }
        }
    } finally { $zip.Dispose() }
    $h = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLower()
    Write-Host "archive: $Archive ($([math]::Round((Get-Item $Archive).Length / 1MB, 1)) MB) sha256 $h"
} finally {
    if ($mapped) { & subst $drv /d | Out-Null }
    $busy.Dispose()
}
