function install_opencl_drivers {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-setup-opencl-cpu-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -q https://www.dropbox.com/s/h0txd04aqfluglq/l_opencl_p_18.1.0.015.tgz || exit $?
    tar -xf l_opencl_p_18.1.0.015.tgz || exit $?
    cd l_opencl_*/ || exit $?

    cat > installer_config.cfg <<EOF
ACCEPT_EULA=accept
CONTINUE_WITH_OPTIONAL_ERROR=yes
PSET_INSTALL_DIR=/opt/intel
CONTINUE_WITH_INSTALLDIR_OVERWRITE=yes
COMPONENTS=DEFAULTS
PSET_MODE=install
INTEL_SW_IMPROVEMENT_PROGRAM_CONSENT=no
SIGNING_ENABLED=yes
EOF
    echo "Running Intel OpenCL driver installer."
    sudo ./install.sh --silent installer_config.cfg
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

install_opencl_drivers || exit $?