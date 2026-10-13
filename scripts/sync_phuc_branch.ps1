$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$branch = 'phuc/baseline-zero-shot'
$remote = 'origin'
$expectedRemote = 'https://github.com/PhucBeossss123/an-y.git'

Push-Location $repo
try {
    $actualBranch = (& git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualBranch -ne $branch) {
        throw "Expected branch '$branch'; found '$actualBranch'. No sync performed."
    }

    $actualRemote = (& git remote get-url $remote).Trim().TrimEnd('/')
    if ($LASTEXITCODE -ne 0 -or $actualRemote -ne $expectedRemote.TrimEnd('/')) {
        throw "Remote '$remote' did not match the expected repository. No sync performed."
    }

    & git fetch $remote "refs/heads/${branch}:refs/remotes/${remote}/${branch}"
    if ($LASTEXITCODE -ne 0) { throw 'git fetch failed. No sync performed.' }

    $counts = @((& git rev-list --left-right --count "HEAD...$remote/$branch").Trim() -split '\s+')
    if ($LASTEXITCODE -ne 0 -or $counts.Count -ne 2) {
        throw 'Could not compare local and remote branch. No sync performed.'
    }
    $ahead = [int]$counts[0]
    $behind = [int]$counts[1]

    if ($ahead -gt 0 -and $behind -gt 0) {
        throw 'Local and remote branches diverged. Resolve and review manually; no push performed.'
    }

    if ($behind -gt 0) {
        $trackedChanges = @(& git status --porcelain --untracked-files=no)
        if ($LASTEXITCODE -ne 0 -or $trackedChanges.Count -gt 0) {
            throw 'Tracked local changes are present. Commit or stash them before pulling; no sync performed.'
        }
        & git merge --ff-only "$remote/$branch"
        if ($LASTEXITCODE -ne 0) { throw 'Fast-forward update failed. No push performed.' }
        $ahead = 0
    }

    if ($ahead -gt 0) {
        & git push $remote "HEAD:refs/heads/$branch"
        if ($LASTEXITCODE -ne 0) { throw 'git push failed.' }
        Write-Output "Pushed $ahead reviewed commit(s) to $remote/$branch."
    }
    else {
        Write-Output "No local commits to push; $branch is synchronized with $remote."
    }
}
finally {
    Pop-Location
}
