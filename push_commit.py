"""Stage, commit, and push a Git update."""

import subprocess as sp


def push():
    """Stage, commit, and push an edit."""
    files = []
    for i in sp.check_output(["git", "status"]).decode().split("\n"):
        nf = "#\tnew file:"
        mf = "#\tmodified:"
        if i[: len(nf)] == nf or i[: len(mf)] == mf:
            f = i.split(" ")[-1]
            files.append(f)
    files = list(set(files))  # Remove duplicates

    print("Committing these files: {}".format(files))

    # Run all py scripts through black for formatting.
    for f in files:
        if f[-3:] == ".py":
            sp.call(["black", f])

    [sp.call(["git", "add", "{}".format(i)]) for i in files]

    commit_message = str(input("Enter commit message:\n"))
    commit_message = "Updated" if commit_message == "" else commit_message
    print("Committing with commit message of: {}\n\n".format(commit_message))
    sp.call(["git", "commit", "-m", "{}".format(commit_message)])
    sp.call(["git", "push"])


push()
