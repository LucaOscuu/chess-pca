from chess_pca.pipeline import run_unified_analysis

if __name__ == "__main__":
    # Metti i PGN dentro: data/raw/ (anche con sottocartelle)
    run_unified_analysis(
        input_path="data/raw",
        max_games=10000,
        min_moves=10,
        cache_dir="data/cache",
        out_dir="data/outputs",
        force_recompute=False,   # metti True se cambi features e vuoi rigenerare cache
        random_state=42,
        max_moves_traj=50,
        min_obs_traj=50,
        run_h1=True,
        h1_m_null=500,
        run_h2=True,
        h2_n_bootstrap=80,
        h2_sample_fraction=0.8,
        run_h3=True,
        h3_start_move=5,
        h3_exclude_resignation=False,
        h3_bins_main=50,
        h3_save_prefix="h3",

     )

