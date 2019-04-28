"""Stage, commit, and push a Git update."""


import subprocess as sp
import argparse as ap


def main():
    """Run it."""
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter,
                               description='''Make a run happen.''')

    parser.add_argument('-p', '--push',
                        action='store_true',
                        help='Push an edit.')

    parser.add_argument('-d', '--pull',
                        action='store_true',
                        help='Pull edits down.')

    parser.add_argument('-s', '--status',
                        action='store_true',
                        help='Show git status.')

    parser.add_argument('-a', '--add',
                        action='store_true',
                        help='Start tracking a file.')

    args = parser.parse_args()

    if args.push:
        push()

    elif args.pull:
        pull()

    elif args.status:
        status()

    elif args.add:
        add()


def push():
    """Stage, commit, and push an edit."""
    s = sp.check_output(['git', 'status']).decode().split('\n')
    p = [_f for _f in s if _f]

    print("Committing these files:")
    files = []
    # list(set(p)) gets rid of duplicates.
    for i in list(set(p)):
        if i[:1] == '\t':
            f = [_f for _f in i[1:].split(' ') if _f][-1]
            files.append(f)
            print(f)

    [sp.call(['git', 'add', '{}'.format(i)]) for i in files]
    sp.call(['git', 'add', '*.py'])

    commit_message = str(input('Enter commit message:\n'))
    commit_message = 'Updated' if commit_message == '' else commit_message
    print("Committing with commit message of: ", commit_message)
    print("\n\n")
    sp.call(['git', 'commit', '-m', '{}'.format(commit_message)])
    sp.call(['git', 'push'])


def pull():
    """Pull some stuff.

    This isn't really worth having be it's own function but whatever.
    """
    sp.call(['git', 'pull'])


def status():
    """Show git status.

    This isn't really worth having be it's own function but whatever.
    """
    sp.call(['git', 'status'])


def add():
    """Pull some stuff.

    This isn't really worth having be it's own function but whatever.
    """
    new_file = eval(input('name of file to be added:\n'))
    sp.call(['git', 'add', '{}'.format(new_file)])


if __name__ == '__main__':
    main()

# The End
