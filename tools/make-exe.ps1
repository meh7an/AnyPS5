param(
    [Parameter(Position = 0)][string]$GamePath,
    [string]$Output,
    [ValidateSet('auto', 'intel', 'amd')][string]$Cpu = 'auto',
    [switch]$Release,
    [switch]$NoConsole
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repository = 'boykopovar/AnyPS5'
$ModuleDirectories = @('sce_module', 'sce_modules', 'prx')
$GuestSuffix = '.guest.prx'
$RuntimeDlls = @('libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll')

function Fail([string]$message) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

function Select-GameFolder {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = 'Select the dumped PS5 game folder (the one containing eboot.bin)'
    $dialog.ShowNewFolderButton = $false
    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { Fail 'No game folder selected.' }
    return $dialog.SelectedPath
}

function Find-GameRoot([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { Fail "Path not found: $path" }
    $item = Get-Item -LiteralPath $path
    if (-not $item.PSIsContainer) {
        if ($item.Name -ne 'eboot.bin') { Fail "Expected a game folder or its eboot.bin, got: $path" }
        return $item.Directory.FullName
    }
    if (Test-Path -LiteralPath (Join-Path $item.FullName 'eboot.bin')) { return $item.FullName }
    $candidates = @(Get-ChildItem -LiteralPath $item.FullName -Directory | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'eboot.bin') })
    if ($candidates.Count -eq 1) { return $candidates[0].FullName }
    if ($candidates.Count -gt 1) { Fail "Several game folders found in $path. Select one of: $(($candidates | ForEach-Object Name) -join ', ')" }
    Fail "eboot.bin not found in $path"
}

function Get-ExecutableKind([string]$path) {
    $stream = [System.IO.File]::OpenRead($path)
    try {
        $magic = New-Object byte[] 4
        $read = $stream.Read($magic, 0, 4)
    } finally {
        $stream.Dispose()
    }
    if ($read -ne 4) { return 'other' }
    $value = [BitConverter]::ToString($magic)
    if ($value -eq '7F-45-4C-46') { return 'elf' }
    if ($value -eq '4F-15-3D-1D' -or $value -eq '54-14-F5-EE') { return 'self' }
    return 'other'
}

function Convert-SelfToElf([string]$source, [string]$destination) {
    $self = [System.IO.File]::ReadAllBytes($source)
    $name = Split-Path $source -Leaf
    $entryCount = [BitConverter]::ToUInt16($self, 0x18)
    $elfOffset = 0x20 + $entryCount * 0x20
    if ($self.Length -lt $elfOffset + 0x40 -or [BitConverter]::ToUInt32($self, $elfOffset) -ne 0x464c457f) { Fail "$name has no embedded ELF header." }
    $segments = @{}
    for ($index = 0; $index -lt $entryCount; $index++) {
        $entry = 0x20 + $index * 0x20
        $properties = [BitConverter]::ToInt64($self, $entry)
        if ($properties -band 0x2) { Fail "$name is encrypted. Dump the game again with executable decryption enabled." }
        if ($properties -band 0x8) { Fail "$name contains compressed segments, which are not supported." }
        if (($properties -shr 11) -band 1) {
            $segments[[int](($properties -shr 20) -band 0xfff)] = @{
                Offset = [BitConverter]::ToInt64($self, $entry + 8)
                Size = [BitConverter]::ToInt64($self, $entry + 16)
            }
        }
    }
    $programHeaderOffset = [BitConverter]::ToInt64($self, $elfOffset + 0x20)
    $programHeaderSize = [BitConverter]::ToUInt16($self, $elfOffset + 0x36)
    $programHeaderCount = [BitConverter]::ToUInt16($self, $elfOffset + 0x38)
    $headersSize = $programHeaderOffset + $programHeaderSize * $programHeaderCount
    $outputSize = $headersSize
    $headers = @()
    for ($index = 0; $index -lt $programHeaderCount; $index++) {
        $header = $elfOffset + $programHeaderOffset + $index * $programHeaderSize
        $programHeader = @{
            Type = [BitConverter]::ToUInt32($self, $header)
            Offset = [BitConverter]::ToInt64($self, $header + 0x08)
            FileSize = [BitConverter]::ToInt64($self, $header + 0x20)
        }
        $headers += $programHeader
        if ($programHeader.FileSize -gt 0) { $outputSize = [Math]::Max($outputSize, $programHeader.Offset + $programHeader.FileSize) }
    }
    $elf = New-Object byte[] $outputSize
    [Array]::Copy($self, [long]$elfOffset, $elf, [long]0, [long]$headersSize)
    for ($index = 0; $index -lt $programHeaderCount; $index++) {
        $programHeader = $headers[$index]
        if ($programHeader.FileSize -le 0) { continue }
        if ($segments.ContainsKey($index)) {
            $segment = $segments[$index]
            [Array]::Copy($self, [long]$segment.Offset, $elf, [long]$programHeader.Offset, [long][Math]::Min($segment.Size, $programHeader.FileSize))
        } elseif ($programHeader.Type -eq 0x6fffff01) {
            [Array]::Copy($self, [long]($self.Length - $programHeader.FileSize), $elf, [long]$programHeader.Offset, [long]$programHeader.FileSize)
        }
    }
    [System.IO.File]::WriteAllBytes($destination, $elf)
}

function Get-RelinkerInput([string]$gameRoot, [string]$output) {
    $eboot = Join-Path $gameRoot 'eboot.bin'
    $ebootKind = Get-ExecutableKind $eboot
    if ($ebootKind -eq 'other') { Fail "eboot.bin is neither an ELF nor a SELF file." }
    $moduleFiles = @()
    foreach ($directory in $ModuleDirectories) {
        $path = Join-Path $gameRoot $directory
        if (Test-Path -LiteralPath $path) {
            $moduleFiles += Get-ChildItem -LiteralPath $path -File | Where-Object { -not $_.Name.EndsWith($GuestSuffix) } | ForEach-Object { @{ Directory = $directory; File = $_; Kind = Get-ExecutableKind $_.FullName } }
        }
    }
    if ($ebootKind -eq 'elf' -and -not ($moduleFiles | Where-Object { $_.Kind -eq 'self' })) { return $eboot }

    Write-Host 'Unpacking signed executables ...'
    $prep = Join-Path $output 'prep'
    New-Item -ItemType Directory -Force -Path $prep | Out-Null
    $relinkerInput = Join-Path $prep 'eboot.elf'
    if ($ebootKind -eq 'self') { Convert-SelfToElf $eboot $relinkerInput } else { Copy-Item -LiteralPath $eboot -Destination $relinkerInput -Force }
    foreach ($module in $moduleFiles) {
        if ($module.Kind -eq 'other') { continue }
        $directory = Join-Path $prep $module.Directory
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
        $destination = Join-Path $directory $module.File.Name
        if ($module.Kind -eq 'self') { Convert-SelfToElf $module.File.FullName $destination } else { Copy-Item -LiteralPath $module.File.FullName -Destination $destination -Force }
    }
    $systemDirectory = Join-Path $gameRoot 'sce_sys'
    if (Test-Path -LiteralPath $systemDirectory) { New-Link $systemDirectory (Join-Path $prep 'sce_sys') $true }
    return $relinkerInput
}

function Get-GameTitle([string]$gameRoot) {
    $paramPath = Join-Path $gameRoot 'sce_sys\param.json'
    $title = $null
    if (Test-Path -LiteralPath $paramPath) {
        try {
            $param = [System.IO.File]::ReadAllText($paramPath) | ConvertFrom-Json
            $localized = $param.localizedParameters
            if ($localized) {
                $language = $localized.defaultLanguage
                if (-not $language) { $language = 'en-US' }
                $title = $localized.$language.titleName
            }
        } catch {
            $title = $null
        }
    }
    if (-not $title) { $title = Split-Path $gameRoot -Leaf }
    $invalid = [System.IO.Path]::GetInvalidFileNameChars()
    $clean = -join ($title.ToCharArray() | Where-Object { [int]$_ -ge 32 -and [int]$_ -lt 127 -and $invalid -notcontains $_ })
    $clean = ($clean -replace '\s+', ' ').Trim(' ', '.')
    if (-not $clean) { $clean = 'game' }
    return $clean
}

function Find-MinGwBin {
    $directories = @()
    foreach ($name in @('g++.exe', 'gcc.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $directories += Split-Path $command.Source -Parent }
    }
    $directories += 'C:\winlibs\mingw64\bin'
    foreach ($directory in $directories) {
        if (Test-Path -LiteralPath (Join-Path $directory $RuntimeDlls[0])) { return $directory }
    }
    return $null
}

function Get-LocalTools {
    $repositoryRoot = Split-Path $PSScriptRoot -Parent
    $relinker = Join-Path $repositoryRoot 'build\core\relinker\relinker.exe'
    if (-not (Test-Path -LiteralPath $relinker)) { return $null }
    $libsDirectory = Join-Path $repositoryRoot 'build\core\libs\libs'
    $libraries = @(Get-ChildItem -LiteralPath $libsDirectory -Filter '*.prx' -File -ErrorAction SilentlyContinue)
    if ($libraries.Count -eq 0) { Fail "Found a local relinker build but no system libraries. Run: cmake --build build --target libs --parallel (or pass -Release to download them)." }
    $files = @($libraries | ForEach-Object FullName)
    $searchPath = @()
    $minGw = Find-MinGwBin
    if ($minGw) {
        $files += $RuntimeDlls | ForEach-Object { Join-Path $minGw $_ } | Where-Object { Test-Path -LiteralPath $_ }
        $searchPath += $minGw
    } else {
        Write-Host "WARNING: MinGW runtime DLLs ($($RuntimeDlls -join ', ')) not found; the game may fail to start." -ForegroundColor Yellow
    }
    return @{ Source = "local build ($repositoryRoot\build)"; Relinker = $relinker; Files = $files; SearchPath = $searchPath }
}

function Get-ReleaseTools {
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
    $headers = @{ 'User-Agent' = 'AnyPS5-make-exe' }
    try {
        $releaseInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repository/releases/latest" -Headers $headers
    } catch {
        Fail "Could not reach GitHub to download AnyPS5: $($_.Exception.Message)"
    }
    $tag = $releaseInfo.tag_name
    $cache = Join-Path $env:LOCALAPPDATA "AnyPS5\releases\$tag"
    $relinker = Join-Path $cache 'relinker.exe'
    $libsDirectory = Join-Path $cache 'libs'
    if (-not (Test-Path -LiteralPath $relinker) -or -not (Test-Path -LiteralPath $libsDirectory)) {
        Write-Host "Downloading AnyPS5 $tag ..."
        New-Item -ItemType Directory -Force -Path $cache | Out-Null
        $relinkerAsset = $releaseInfo.assets | Where-Object name -eq "relinker-$tag.exe"
        $libsAsset = $releaseInfo.assets | Where-Object name -eq "prx-windows-$tag.zip"
        if (-not $relinkerAsset -or -not $libsAsset) { Fail "Release $tag does not contain the Windows relinker and libraries." }
        $archive = Join-Path $cache 'libs.zip'
        Invoke-WebRequest -UseBasicParsing -Uri $libsAsset.browser_download_url -Headers $headers -OutFile $archive
        $extracted = Join-Path $cache 'extract'
        if (Test-Path -LiteralPath $extracted) { Remove-Item -LiteralPath $extracted -Recurse -Force }
        Expand-Archive -LiteralPath $archive -DestinationPath $extracted
        Move-Item -LiteralPath (Join-Path $extracted 'libs') -Destination $libsDirectory
        Remove-Item -LiteralPath $extracted -Recurse -Force
        Remove-Item -LiteralPath $archive -Force
        Invoke-WebRequest -UseBasicParsing -Uri $relinkerAsset.browser_download_url -Headers $headers -OutFile $relinker
    }
    $files = @(Get-ChildItem -LiteralPath $libsDirectory -File | ForEach-Object FullName)
    return @{ Source = "release $tag"; Relinker = $relinker; Files = $files; SearchPath = @($libsDirectory) }
}

function Test-Intel {
    if ($Cpu -eq 'intel') { return $true }
    if ($Cpu -eq 'amd') { return $false }
    return $env:PROCESSOR_IDENTIFIER -match 'GenuineIntel'
}

function New-Link([string]$source, [string]$destination, [bool]$directory) {
    if (Get-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue) { return }
    $ErrorActionPreference = 'Continue'
    if ($directory) {
        cmd.exe /d /c mklink /J $destination $source 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { Fail "Could not link folder $source. Put the output on an NTFS drive with -Output." }
        return
    }
    cmd.exe /d /c mklink /H $destination $source 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { Copy-Item -LiteralPath $source -Destination $destination }
}

function Build-App0([string]$gameRoot, [string]$app0) {
    New-Item -ItemType Directory -Force -Path $app0 | Out-Null
    foreach ($entry in Get-ChildItem -LiteralPath $gameRoot -Force) {
        $destination = Join-Path $app0 $entry.Name
        if ($entry.PSIsContainer -and $ModuleDirectories -contains $entry.Name) {
            New-Item -ItemType Directory -Force -Path $destination | Out-Null
            foreach ($module in Get-ChildItem -LiteralPath $entry.FullName -Force) {
                if ($module.Name.EndsWith($GuestSuffix)) { continue }
                New-Link $module.FullName (Join-Path $destination $module.Name) $module.PSIsContainer
            }
        } else {
            New-Link $entry.FullName $destination $entry.PSIsContainer
        }
    }
}

if (-not $GamePath) { $GamePath = Select-GameFolder }
$gameRoot = Find-GameRoot $GamePath
if (-not ($ModuleDirectories | Where-Object { Test-Path -LiteralPath (Join-Path $gameRoot $_) })) {
    Fail "No sce_module, sce_modules or prx folder next to eboot.bin in $gameRoot"
}

if (-not $Output) { $Output = Join-Path (Split-Path $gameRoot -Parent) "$(Split-Path $gameRoot -Leaf)-PC" }
$Output = [System.IO.Path]::GetFullPath($Output)
$gameRootFull = [System.IO.Path]::GetFullPath($gameRoot).TrimEnd('\')
if ($Output.TrimEnd('\') -eq $gameRootFull -or $Output.StartsWith("$gameRootFull\", [System.StringComparison]::OrdinalIgnoreCase)) {
    Fail 'The output folder must be outside the game folder.'
}

$tools = $null
if (-not $Release) { $tools = Get-LocalTools }
if (-not $tools) { $tools = Get-ReleaseTools }

$title = Get-GameTitle $gameRoot
$executable = Join-Path $Output "$title.exe"
$intel = Test-Intel

Write-Host "Game:     $gameRoot"
Write-Host "Output:   $executable"
Write-Host "AnyPS5:   $($tools.Source)"
Write-Host "CPU:      $(if ($intel) { 'Intel (--to-intel)' } else { 'AMD' })"
Write-Host ''

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$eboot = Get-RelinkerInput $gameRoot $Output
$arguments = @('--windows')
if ($intel) { $arguments += '--to-intel' }
if ($NoConsole) { $arguments += '--windows-gui' }
$arguments += @($eboot, $executable)

$env:PATH = (@($tools.SearchPath) + $env:PATH) -join ';'
& $tools.Relinker @arguments
if ($LASTEXITCODE -ne 0) { Fail "Relinker failed (exit code $LASTEXITCODE). This game is probably not supported yet; see docs/user/COMPATIBILITY.md." }

$libs = Join-Path $Output 'libs'
New-Item -ItemType Directory -Force -Path $libs | Out-Null
foreach ($file in $tools.Files) { Copy-Item -LiteralPath $file -Destination $libs -Force }

Build-App0 $gameRoot (Join-Path $Output 'app0')

Write-Host ''
Write-Host "Done. Start the game with: $executable" -ForegroundColor Green
