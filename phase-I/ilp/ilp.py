fractions = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

for fraction in fractions:

    model, x_vars, y_vars = build_core_model(
        dataset,
        programmable_fraction=fraction,
    )

    status = solve_model(model)

    if status == GRB.OPTIMAL:

        selected = extract_switch_selection(
            x_vars,
            sid_to_name,
        )

        print(
            f"{fraction * 100:.0f}% programmable: "
            f"{len(selected)} switches selected"
        )

    elif status == GRB.INFEASIBLE:

        print(
            f"{fraction * 100:.0f}% programmable: "
            "INFEASIBLE"
        )
