<!-- omit in toc -->
# Contributing to Neuralmag

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions.

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->
## Table of Contents

- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
    - [Cloning the Repository](#cloning-the-repository)
    - [Installing the Development Version](#installing-the-development-version)
    - [Next Steps](#next-steps)
  - [Improving The Documentation](#improving-the-documentation)
- [Styleguides](#styleguides)
  - [Pre-commit Guidelines](#pre-commit-guidelines)
    - [Setting up Pre-commit](#setting-up-pre-commit)
    - [Our Hooks](#our-hooks)
    - [Best Practices](#best-practices)
- [Join The Project Team](#join-the-project-team)



## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation]().

Before you ask a question, it is best to search for existing [Issues](https://gitlab.com/neuralmag/neuralmag/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://gitlab.com/neuralmag/neuralmag/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (torch, torchdiffeq, etc), depending on what seems relevant.

We will then take care of the issue as soon as possible.

<!--
You might want to create a separate issue tag for questions and include it in this description. People should then tag their issues accordingly.

Depending on how large the project is, you may want to outsource the questioning, e.g. to Stack Overflow or Gitter. You may add additional contact and information possibilities:
- IRC
- Slack
- Gitter
- Stack Overflow tag
- Blog
- FAQ
- Roadmap
- E-Mail List
- Forum
-->

## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs

<!-- omit in toc -->
#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://neuralmag.gitlab.io/neuralmag/). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [issues](https://gitlab.com/neuralmag/neuralmag/-/issues).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

<!-- omit in toc -->
#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to <>.
<!-- You may add a PGP key to allow the messages to be sent encrypted as well. -->

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://gitlab.com/neuralmag/neuralmag/issues/new). (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

<!-- You might want to create an issue template for bugs and errors that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Neuralmag, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

<!-- omit in toc -->
#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://neuralmag.gitlab.io/neuralmag/) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://gitlab.com/neuralmag/neuralmag/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

<!-- omit in toc -->
#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://gitlab.com/neuralmag/neuralmag/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux. <!-- this should only be included if the project has a GUI -->
- **Explain why this enhancement would be useful** to most Neuralmag users. You may also want to point out the other projects that solved it better and which could serve as inspiration.

<!-- You might want to create an issue template for enhancement suggestions that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->

### Your First Code Contribution
To contribute to neuralmag, you'll need to set up your development environment. Follow these steps to get started:

#### Cloning the Repository

You can clone the repository using either SSH or HTTPS, depending on your setup:

- **SSH** (recommended if you have an SSH key configured):
  ```bash
  git clone git@gitlab.com:neuralmag/neuralmag.git
  ```

- **HTTPS** (use this method if you don't have an SSH key):
  ```bash
  git clone https://gitlab.com/neuralmag/neuralmag.git
  ```

#### Installing the Development Version

After cloning, navigate into the cloned directory:
```bash
cd neuralmag
```

Install the development dependencies along with the package in editable mode:
```bash
pip install -e .[dev]
```

This setup will allow you to make changes to the codebase and test them in real-time.

#### Next Steps

Once installed, you can begin making changes and testing them using the tools provided in the development environment. Remember to keep your branch updated and adhere to the coding and contribution guidelines provided.

<!-- TODO
include Setup of env, IDE and typical getting started instructions?

-->

### Improving The Documentation
<!-- TODO
Updating, improving and correcting the documentation

-->

## Styleguides
### Pre-commit Guidelines

We use pre-commit hooks to ensure code quality and consistency.

#### Setting up Pre-commit

1. **Install pre-commit**:
   Pre-commit is installed when installing the development version of neuralmag.

2. **Install the pre-commit hooks**:
   Run the following command in the repository root to set up the pre-commit hooks:
   ```bash
   pre-commit install
   ```
   This will enable pre-commits to run automatically whenever you make a commit.

3. **Manual Run**:
   To manually run pre-commit on all files in the repository, use:
   ```bash
   pre-commit run --all-files
   ```

#### Our Hooks

- **Code Formatting**: We use `Black` for Python files and Jupyter notebooks, ensuring consistent coding styles.
- **Import Sorting**: `isort` is used to sort imports in a standard way across all Python files.
- **Syntax and Debug Checks**:
  - `check-merge-conflict`: Prevents code with merge conflict markers from being committed.
  - `check-toml`: Ensures TOML files are correctly formatted and syntactically correct.
  - `debug-statements`: Checks for any debug-related statements in Python source files.

#### Best Practices

Ensure you have activated pre-commit locally and all hooks pass before pushing your changes. This saves time and streamlines the review process, making it easier for everyone involved.

For additional information or troubleshooting with pre-commit, visit the [official pre-commit documentation](https://pre-commit.com/).
```

## Join The Project Team
<!-- TODO -->

<!-- omit in toc -->
## Attribution
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
