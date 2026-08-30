# Era 1: Foundation, Microsoft Extraction & Testing Architecture (2022)

## Overview
In early 2022, `pandas-stubs` was spun out from Microsoft's `python-type-stubs` repository into the `pandas-dev` organization under PEP 561 as a standalone, community-driven stub repository.

## Key Maintainer Debates & Breakthroughs

1. **Test Modernization**: `Dr-Irv` eradicated legacy runtime inspection in PR pandas-dev/pandas-stubs#7: use assert_type instead of check_XXX in some tests (pandas-dev/pandas-stubs@17b4423e65abcf8a0b403bf9fe445f06d3ff2236) and PR pandas-dev/pandas-stubs#8: use assert_type throughout, remove check_ functions (pandas-dev/pandas-stubs@51aeba53ceb106492a167d87dd117c2922c5d147), introducing `assert_type()` across all test modules.
2. **Nightly Testing Pipeline**: `twoertwein` established nightly CI runs against upstream pandas dev builds in PR pandas-dev/pandas-stubs#238: run pytest against nightly (pandas-dev/pandas-stubs@b0728425368ed472f4812995d10bace7fa560e20) (46 discussion threads).
3. **Scalar Expansion**: `bashtage` executed foundational overhauls of pandas scalars: `Timedelta` (PR pandas-dev/pandas-stubs#388: ENH: Improve typing for Timedelta (pandas-dev/pandas-stubs@5df1c515c8f8b14d402bd5dee0aaa8503074e4fb)), `Timestamp` (PR pandas-dev/pandas-stubs#389: ENH: Improve typing for Timestamp (pandas-dev/pandas-stubs@56bcec7df68bf41bc12136a7cb1166c80d5d5be6)), `Period` (PR pandas-dev/pandas-stubs#390: ENH: Improve typing for Period (pandas-dev/pandas-stubs@69710a158023631b6ee5aafe2c40b94fe9b0262d)), and `Styler` (PR pandas-dev/pandas-stubs#282: ENH: Improve Styler typing (pandas-dev/pandas-stubs@ad517c74050f6721469aeba1cdf8ce1a8d5f277e)).
4. **Python 3.11 Support**: `KotlinIsland` and contributors added Python 3.11 typing features in PR pandas-dev/pandas-stubs#398: (🎁) Support Python 3.11.

## Key PRs
- pandas-dev/pandas-stubs#6: clean up with black and tests.  Test PRs (pandas-dev/pandas-stubs@1af190411028fd05f7fceaa7043a5c811d864e33)
- pandas-dev/pandas-stubs#7: use assert_type instead of check_XXX in some tests (pandas-dev/pandas-stubs@17b4423e65abcf8a0b403bf9fe445f06d3ff2236)
- pandas-dev/pandas-stubs#8: use assert_type throughout, remove check_ functions (pandas-dev/pandas-stubs@51aeba53ceb106492a167d87dd117c2922c5d147)
- pandas-dev/pandas-stubs#10: test the stubs with mypy, pyright (pandas-dev/pandas-stubs@63d03bc9297357715eda7a41b8f694b91b51395e)
- pandas-dev/pandas-stubs#24: Nopytyped (pandas-dev/pandas-stubs@2eba4d4e512421927a4ba2e6d0ac7bbd4e934afe)
- pandas-dev/pandas-stubs#39: Fix to_dict and from_dict type stubs (pandas-dev/pandas-stubs@8fbe101a4b28335cac7391d3630288553e01ed5b)
- pandas-dev/pandas-stubs#50: CI/CD Update
- pandas-dev/pandas-stubs#59: TYP/CI: enable more pyright checks (pandas-dev/pandas-stubs@1118b791e4cee09bf3129ad216c658a3c6dc9df0)
- pandas-dev/pandas-stubs#65: added type hint to read_csv usecols (pandas-dev/pandas-stubs@a4860495b0a7852719ad7503899d5a10befd8066)
- pandas-dev/pandas-stubs#83: CI: run style checks on CI (pandas-dev/pandas-stubs@3c7e0f65b6c8c78b8095ae8435e7bd1f7102f4c4)
- pandas-dev/pandas-stubs#106: Allow `num` to be a `complex` type to support `Series` operations. (pandas-dev/pandas-stubs@9d80790c5bde23d597663eee7d5f5a3cfbbbde6b)
- pandas-dev/pandas-stubs#114: assert types at runtime (pandas-dev/pandas-stubs@2fd9697fe7f75e54845c0926f22a6c2df6d9f219)
- pandas-dev/pandas-stubs#124: Allow subtypes of List[Scalar] by introducing ScalarArg (pandas-dev/pandas-stubs@d2f32648340c29bf57f73753f06ab9c6ad18a98a)
- pandas-dev/pandas-stubs#130: Annotate Series `to_dict` and `to_list` with generics (pandas-dev/pandas-stubs@a3fdd9c1d80cfd1c0535718b7165548be01b7617)
- pandas-dev/pandas-stubs#148: groupby.__iter__() fix types (pandas-dev/pandas-stubs@a6dd774bcb0cb43f209dd88e5adee05998824dd8)
- pandas-dev/pandas-stubs#151: TYP: future annotations (pandas-dev/pandas-stubs@9383884b819b24c83e6757cade8800b15062b6c9)
- pandas-dev/pandas-stubs#166: CLEAN: Align Groupby types (pandas-dev/pandas-stubs@927d4388775c829859e5caf4600b2f8ecf8e190d)
- pandas-dev/pandas-stubs#173: Standardised aggregate functions typing (pandas-dev/pandas-stubs@8f9ba75f595b434987454881e8e016669ab45100)
- pandas-dev/pandas-stubs#174: Fixed typing on IntervalIndex functions (pandas-dev/pandas-stubs@a8bc6c63a66f984c4163e491e0b202bbcb2f1c6d)
- pandas-dev/pandas-stubs#177: More specific types for GroupBy.apply. (pandas-dev/pandas-stubs@02e1748becb97e485da6930ab4ed9fea382d8ed9)
- pandas-dev/pandas-stubs#183: stubtest: option to not set ignore-missing-stub (pandas-dev/pandas-stubs@1ec0bb9f5dd4714e50143e1067e3b6addba6cd78)
- pandas-dev/pandas-stubs#190: Added missing groupby methods and made SeriesGroupBy generic (pandas-dev/pandas-stubs@fed3be4c53250ad749f3f78ce7831bb6b27f909c)
- pandas-dev/pandas-stubs#238: run pytest against nightly (pandas-dev/pandas-stubs@b0728425368ed472f4812995d10bace7fa560e20)
- pandas-dev/pandas-stubs#282: ENH: Improve Styler typing (pandas-dev/pandas-stubs@ad517c74050f6721469aeba1cdf8ce1a8d5f277e)
- pandas-dev/pandas-stubs#317: MAINT: Bump pandas to 1.5.0 (pandas-dev/pandas-stubs@d1f00f3c1576a9e64f9729c7daa7612fcfa0ed63)
- pandas-dev/pandas-stubs#355: ENH: Improve typing of some general functions (pandas-dev/pandas-stubs@d2ae5ee5f0866b111e33a0b471a48d0bef4cc283)
- pandas-dev/pandas-stubs#378: added_int_bitwise_operator (pandas-dev/pandas-stubs@c5d66489a6de952a4ae8c3fc313da0f560578338)
- pandas-dev/pandas-stubs#383: ENH: Improve Pandas scalars
- pandas-dev/pandas-stubs#388: ENH: Improve typing for Timedelta (pandas-dev/pandas-stubs@5df1c515c8f8b14d402bd5dee0aaa8503074e4fb)
- pandas-dev/pandas-stubs#389: ENH: Improve typing for Timestamp (pandas-dev/pandas-stubs@56bcec7df68bf41bc12136a7cb1166c80d5d5be6)
- pandas-dev/pandas-stubs#390: ENH: Improve typing for Period (pandas-dev/pandas-stubs@69710a158023631b6ee5aafe2c40b94fe9b0262d)
- pandas-dev/pandas-stubs#398: (🎁) Support Python 3.11
- pandas-dev/pandas-stubs#432: added np.timedelta64 for series arithmetic methods (pandas-dev/pandas-stubs@b7163c25f2b1a986078e3787c5110913054088f0)
