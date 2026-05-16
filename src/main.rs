use llmb::cli::{Cli, Commands, ModelsCommands};
use clap::Parser;
use colored::Colorize;

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Init { config } => llmb::cli::cmd_init(&config),
        Commands::Models { command } => match command {
            ModelsCommands::List { config } => llmb::cli::cmd_models_list(&config),
            ModelsCommands::Fetch { config } => llmb::cli::cmd_models_fetch(&config),
        },
        Commands::Bench { config, out, devices, models, runs } => {
            llmb::cli::cmd_bench_run(&config, out, &devices, &models, runs)
        }
        Commands::Report { dir } => llmb::cli::cmd_report_open(dir),
        Commands::Setup { force } => llmb::cli::cmd_setup(force),
        Commands::Doctor { config } => llmb::cli::cmd_doctor(&config),
        Commands::Compare { results, out } => llmb::cli::cmd_compare(&results, &out),
    };

    if let Err(e) = result {
        eprintln!("{} {}", "error:".red().bold(), e);
        std::process::exit(1);
    }
}
